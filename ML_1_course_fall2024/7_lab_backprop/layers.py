import numpy as np
import scipy as sp
import scipy.signal
import skimage

class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    Moreover, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        input_grad = module.backward(input, output_grad)
    """
    def __init__ (self):
        self._output = None
        self._input_grad = None
        self.training = True
    
    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        self._output = self._compute_output(input)
        return self._output

    def backward(self, input, output_grad):
        """
        Performs a backpropagation step through the module, with respect to the given input.
        
        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self._input_grad = self._compute_input_grad(input, output_grad)
        self._update_parameters_grad(input, output_grad)
        return self._input_grad
    

    def _compute_output(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which will be stored in the `_output` field.

        Example: in case of identity operation:
        
        output = input 
        return output
        """
        raise NotImplementedError
        

    def _compute_input_grad(self, input, output_grad):
        """
        Returns the gradient of the module with respect to its own input. 
        The shape of the returned value is always the same as the shape of `input`.
        
        Example: in case of identity operation:
        input_grad = output_grad
        return input_grad
        """
        
        raise NotImplementedError
    
    def _update_parameters_grad(self, input, output_grad):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass
    
    def zero_grad(self): 
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass
        
    def get_parameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
        
    def get_parameters_grad(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
    
    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"
    
class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.99):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = 0.
        self.moving_variance = 1.

    def updateMeanVariance(self, batch_mean, batch_variance):
        self.moving_mean = batch_mean if self.moving_mean is None else self.moving_mean
        self.moving_variance = batch_variance if self.moving_variance is None else self.moving_variance

        # moving_mean update
        np.multiply(self.moving_mean, self.alpha, out=self.moving_mean)
        np.multiply(batch_mean, 1-self.alpha, out=batch_mean)
        np.add(self.moving_mean, batch_mean, out=self.moving_mean)
        # moving_variance
        np.multiply(self.moving_variance, self.alpha, out=self.moving_variance)
        np.multiply(batch_variance, 1-self.alpha, out=batch_variance)
        np.add(self.moving_variance, batch_variance, out=self.moving_variance)

    def _compute_output(self, input):
        batch_mean = np.mean(input, axis=0) if self.training else self.moving_mean
        batch_variance = np.var(input, axis=0) if self.training else self.moving_variance

        self.output = (input - batch_mean) / np.sqrt(batch_variance + self.EPS)
        if self.training:
            self.updateMeanVariance(batch_mean, batch_variance)
        return self.output

    def _compute_input_grad(self, input, gradOutput):
        batch_mean = np.mean(input, axis=0) if self.training else self.moving_mean
        batch_variance = np.var(input, axis=0) if self.training else self.moving_variance
        m = input.shape[0]

        variable0 = input - batch_mean
        variable1 = np.sum(gradOutput * variable0, axis=0)
        variable2 = np.sum(gradOutput, axis=0)
        variable3 = np.sqrt(batch_variance + self.EPS)

        self.grad_input = gradOutput / variable3
        self.grad_input -= variable1 * variable0 / m / variable3 / (batch_variance + self.EPS)
        self.grad_input -= variable2 / m / variable3
        self.grad_input += variable1 * np.sum(variable0, 0) / m**2 / variable3**(3/2)

        return self.grad_input

    def __repr__(self):
        return "BatchNormalization"
    
class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def _compute_output(self, input):
        output = input * self.gamma + self.beta
        return output
        
    def _compute_input_grad(self, input, output_grad):
        grad_input = output_grad * self.gamma
        return grad_input
    
    def _update_parameters_grad(self, input, output_grad):
        self.gradBeta = np.sum(output_grad, axis=0)
        self.gradGamma = np.sum(output_grad*input, axis=0)
    
    def zero_grad(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def get_parameters(self):
        return [self.gamma, self.beta]
    
    def get_parameters_grad(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"
    
class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p, self.mask = p, None

    def _compute_output(self, input):
        if not self.training:
            self.output = input
            return self.output
        self.mask = np.random.choice(2, input.shape, p=[self.p, (1.0 - self.p)])
        self.output = np.multiply(input, self.mask) / (1 - self.p)
        return self.output

    def _compute_input_grad(self, input, gradOutput):
        self.grad_input = np.multiply(gradOutput, self.mask) / (1 - self.p)
        return self.grad_input

    def __repr__(self):
        return "Dropout"
    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size

        stdv = 1./np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size = (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def _compute_output(self, input):
        batch_size, in_channels, h, w = input.shape
        pad_size = self.kernel_size // 2
        padded_input = np.pad(input, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
        output = np.zeros((batch_size, self.out_channels, h, w), dtype=np.float32)  # Явное указание типа данных

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    correlation_result = scipy.signal.correlate(padded_input[b, in_c], self.W[out_c, in_c], mode='valid')
                    output[b, out_c] += correlation_result.astype(np.float32) # Явное преобразование типа
                output[b, out_c] += self.b[out_c].astype(np.float32) # Явное преобразование типа

        self._output = output.astype(np.float32)  # Явное преобразование типа
        return self._output

    def _compute_input_grad(self, input, gradOutput):
        batch_size, out_channels, h_out, w_out = gradOutput.shape
        _, in_channels, h_in, w_in = input.shape
        pad_size = self.kernel_size // 2
        padded_gradOutput = np.pad(gradOutput, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')
        input_grad = np.zeros((batch_size, self.in_channels, h_in, w_in))
        flipped_W = np.flip(self.W, axis=(2, 3))

        for b in range(batch_size):
            for in_c in range(self.in_channels):
                for out_c in range(self.out_channels):
                    input_grad[b, in_c] += scipy.signal.correlate(padded_gradOutput[b, out_c], flipped_W[out_c, in_c], mode='valid')

        self._input_grad = input_grad
        return self._input_grad

    def accGradParameters(self, input, gradOutput):
        batch_size, in_channels, h, w = input.shape
        pad_size = self.kernel_size // 2
        padded_input = np.pad(input, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant')

        for out_c in range(self.out_channels):
            for in_c in range(self.in_channels):
                for b in range(batch_size):
                    self.gradW[out_c, in_c] += scipy.signal.correlate(padded_input[b, in_c], gradOutput[b, out_c], mode='valid')

            self.gradb[out_c] += np.sum(gradOutput[:, out_c])

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' %(s[1],s[0])
        return q
    

import numpy as np
import scipy.signal

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size
       
        stdv = 1. / np.sqrt(in_channels * kernel_size * kernel_size)
        self.W = np.random.uniform(-stdv, stdv, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)                                                                                            
        
    def _compute_output(self, input):
        """
        Compute the forward pass of the convolution operation.
        """
        pad_size = self.kernel_size // 2
        batch_size, in_channels, h, w = input.shape
        
        assert in_channels == self.in_channels, "Input channels mismatch"
        
        # Zero-padding the input
        padded_input = np.pad(input, 
                              pad_width=((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                              mode='constant', 
                              constant_values=0)
        
        # Initialize the output tensor
        output = np.zeros((batch_size, self.out_channels, h, w))
        
        # Perform correlation (or cross-correlation)
        for n in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    output[n, oc] += scipy.signal.correlate(
                        padded_input[n, ic], 
                        self.W[oc, ic], 
                        mode='valid'
                    )
                output[n, oc] += self.b[oc]  # Add bias
        
        self._output = output
        return self._output
    
    def _compute_input_grad(self, input, gradOutput):
        """
        Compute the gradient of the input with respect to the loss during backpropagation.
        """
        pad_size = self.kernel_size // 2
        batch_size, in_channels, h, w = input.shape
        
        # Zero-pad gradOutput
        padded_grad_output = np.pad(gradOutput, 
                                    pad_width=((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                                    mode='constant', 
                                    constant_values=0)
        
        # Initialize the input gradient tensor
        input_grad = np.zeros_like(input)
        
        # Compute the input gradient
        for n in range(batch_size):
            for ic in range(self.in_channels):
                for oc in range(self.out_channels):
                    input_grad[n, ic] += scipy.signal.correlate(
                        padded_grad_output[n, oc], 
                        self.W[oc, ic][::-1, ::-1],  # Flip the kernel
                        mode='valid'
                    )
        
        self._input_grad = input_grad
        return self._input_grad
    
    def accGradParameters(self, input, gradOutput):
        """
        Accumulate the gradients with respect to the weights and biases.
        """
        pad_size = self.kernel_size // 2
        batch_size, in_channels, h, w = input.shape
        
        # Zero-pad input
        padded_input = np.pad(input, 
                              pad_width=((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                              mode='constant', 
                              constant_values=0)
        
        # Compute weight gradients
        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                for n in range(batch_size):
                    self.gradW[oc, ic] += scipy.signal.correlate(
                        padded_input[n, ic], 
                        gradOutput[n, oc],  # Gradients from the next layer
                        mode='valid'
                    )
        
        # Compute bias gradients (sum over batch and all spatial dimensions)
        self.gradb = np.sum(gradOutput, axis=(0, 2, 3))
    
    def zeroGradParameters(self):
        """
        Zero the stored gradients for weights and biases.
        """
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        """
        Return the list of parameters (weights and biases).
        """
        return [self.W, self.b]
    
    def getGradParameters(self):
        """
        Return the list of gradients for parameters.
        """
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        """
        Pretty representation of the Conv2d module.
        """
        s = self.W.shape
        return f'Conv2d {s[1]} -> {s[0]}'