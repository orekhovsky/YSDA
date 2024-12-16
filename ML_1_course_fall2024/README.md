- Женя Соколов курс на степике Auto ML
- Geometrical ML
- [NLP](https://lena-voita.github.io/nlp_course.html)
- [Practical RL](https://github.com/yandexdataschool/Practical_RL)
- ML HEP
- Dataset Search - для поиска датасетов 
- Hugging Face - куча моделей
- Papers with code - узнать про тренды в исследованиях 
- UCI ML repository - маленькие общепринятые датасеты 
- форматер Black 
- Фишеровский подход
- Out of back для маленьких выборок 
 

## Important Links 
[Курс на гитхабе](https://github.com/girafe-ai/ml-course/tree/24f_ysda/week0_01_naive_bayes)

[Учебник от Яндекс по МО](https://education.yandex.ru/handbook/ml)

[SDG](https://education.yandex.ru/handbook/ml/article/shodimost-sgd)

[Реализация  GD, МНК и другое в питоне](https://www.dmitrymakarov.ru/opt/mlr-04-2/#34-funktsiya-khyubera)

## Курс

### Deep learning

<mark style="background: #FFB86CA6;">Основы DL  (7 ноября)</mark>

🎥Видеоматериалы
- [Запись](https://disk.yandex.ru/i/BrXC9_eRhjhnyA)

 
📒Ноутбуки
- [Pytorch, tensor](https://colab.research.google.com/github/girafe-ai/ml-course/blob/23s_harbour/day10_intro_to_DL/day10_pytorch_practice.ipynb)


Статьи/полезные материалы
- [Обзор про софтмакс, статья](https://gonzoml.substack.com/p/make-softmax-great-again)

---
<mark style="background: #FFB86CA6;">RNN (14 ноября)</mark>

🎥Видеоматериалы
- [Лекция про RNN LSTM](https://disk.yandex.ru/i/7UKAG14gM_C5Yw)
- [Практика по RNN](https://disk.yandex.ru/i/7UKAG14gM_C5Yw)


📒Ноутбуки
- [Ноутбук Rnn, seq2seq](https://colab.research.google.com/github/girafe-ai/ml-course/blob/24f_ysda/week0_08_language_modeling/seq2seq_rnn_practice_updated.ipynb#scrollTo=Fx-Qs60ATHbm)

Статьи/полезные материалы
- [Warm uo test](https://docs.google.com/forms/d/e/1FAIpQLSfxf2kCHAkdhCkj1LykmDP1WTJx0L6mzUue1nlXxNG4V2kdTw/viewform)
- Замечательная статья за авторством Andrej Karpathy об использовании RNN: [link](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- Пример char-rnn от Andrej Karpathy: [github repo](https://github.com/karpathy/char-rnn)
- Замечательный пример генерации поэзии Шекспира: [github repo](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb)
- Загрузка текста в модель nano GPT [гитхаб](https://github.com/vak/nanoGPT-bit-tokenizer), [ноутбук](https://colab.research.google.com/drive/1xFDdWu4Xgfz-knUORg5gr-p83gjhCvoy)
---
<mark style="background: #FFB86CA6;">CNN (21 ноября)</mark>

🎥Видеоматериалы
	[Операция свертки. Сверточные нейронные сети. Инвариантность и эквивариантность](https://disk.yandex.ru/i/ObYsVEXxxfUgkA)

📒Ноутбуки
	[Работа с изображениями](https://drive.google.com/file/d/1W5IZUVQENo-vwiHNqlnSliOa_u1NQtES/view?usp=sharing)
	
Статьи/полезные материалы
	[Статья про переобучение в природе, пятно на клюве чайки и прочее](https://habr.com/ru/articles/370541/)
	Статья про то, как видят алгоритмы CNN [GradCam](https://github.com/jacobgil/pytorch-grad-cam)

<mark style="background: #FFB86CA6;">BatchNorm, LayerNorm, Dropout. Проблема затухающего градиента и её решение</mark>

🎥Видеоматериалы
- [запись](https://disk.yandex.ru/i/bIoYfFb0bGsUGA)

📒Ноутбуки
- [Vanishing grad](https://github.com/girafe-ai/ml-course/blob/24f_ysda/week0_10_nn_opt_and_reg/vanishing_grad_example.ipynb)
- [Pytorch dataloaders](https://github.com/girafe-ai/ml-course/blob/24f_ysda/week0_10_nn_opt_and_reg/practice_pytorch_and_dataloaders.ipynb)
- [какой-то скрипт (notmnist.py) ](https://github.com/girafe-ai/ml-course/blob/24f_ysda/week0_10_nn_opt_and_reg/notmnist.py)

📘Домашка
	
Статьи/полезные материалы
- [Оптимизация градиентного шага через pytorch](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
- [Эта же лекция на хендбуке](https://education.yandex.ru/handbook/ml/article/tonkosti-obucheniya)
- [Гипотеза о лотерейном билете](https://habr.com/ru/articles/718748/)


❗Заметки
- Инициализировать веса нулями плохо - все латентные представления становятся похожими друг на друга, получается одно латентное представление, дублированное несколько раз 
- Инициализировать рандомом тоже плохо, если распределения фичей и весов из нормального распределения, то в латнентном представлении дисперсия возрастёт в n раз, где n это количество фичей. Поэтому рандомные веса для инициализации нужно делить на n (в случае линейного слоя)
- В Pytorch по умолчанию для каждого слоя свой инициализатор 

### Self-supervised Learning. Learning to rank. Contrastive learning
<mark style="background: #FFB86CA6;">от 5 декабря</mark>

🎥Видеоматериалы
	[запись](https://disk.yandex.ru/i/opN5layMq3HYwg)
 
📒Ноутбуки
	[Interacting with CLIP](https://colab.research.google.com/drive/1taTqkaCRFNSwb1HUhkw0dtVvrKdUUsK_?usp=sharing)
	
Статьи/полезные материалы
