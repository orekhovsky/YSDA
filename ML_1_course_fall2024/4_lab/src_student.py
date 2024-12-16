import contextlib
import enum
import json
import os
import pathlib
import typing as tp
import uuid
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import mode
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score


class NodeType(enum.Enum):
    REGULAR = 1
    TERMINAL = 2


def gini(y: np.ndarray) -> float:
    """
    Computes Gini index for given set of labels
    :param y: labels
    :return: Gini impurity
    """
    _,counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini_index =1 - np.sum(probabilities**2)

    
    return gini_index


def weighted_impurity(y_left: np.ndarray, y_right: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes weighted impurity by averaging children impurities.
    
    :param y_left: left partition
    :param y_right: right partition
    :return: averaged impurity, left child impurity, right child impurity
    """
    left_impurity = gini(y_left)
    right_impurity = gini(y_right)
    
    total_len = len(y_left) + len(y_right)
    weighted_impurity = (len(y_left) / total_len) * left_impurity + (len(y_right) / total_len) * right_impurity
    return weighted_impurity, left_impurity, right_impurity


def create_split(feature_values: np.ndarray, threshold: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    splits given 1-d array according to relation to threshold into two subarrays
    :param feature_values: feature values extracted from data
    :param threshold: value to compare with
    :return: two sets of indices
    """
    left_idx = np.where(feature_values <= threshold)[0]
    right_idx = np.where(feature_values > threshold)[0]
    
    return left_idx, right_idx


def _best_split(self, X: np.ndarray, y: np.ndarray):
    """
    finds best split
    :param X: Data, passed to node
    :param y: labels
    :return: best feature, best threshold, left child impurity, right child impurity
    """
    lowest_impurity = np.inf
    best_feature_id = None
    best_threshold = None
    lowest_left_child_impurity, lowest_right_child_impurity = None, None
    features = self._meta.rng.permutation(X.shape[1])
    for feature in features:
        current_feature_values = X[:, feature]
        thresholds = np.unique(current_feature_values)
        for threshold in thresholds:
            left_idx, right_idx = create_split(current_feature_values, threshold)
            y_left, y_right = y[left_idx], y[right_idx]
            if len(y_left) == 0 or len(y_right) == 0:
                continue  # Пропустить, если одна сторона пустая
            current_weighted_impurity, current_left_impurity, current_right_impurity = weighted_impurity(y_left, y_right)
            if current_weighted_impurity < lowest_impurity:
                lowest_impurity = current_weighted_impurity
                best_feature_id = feature
                best_threshold = threshold
                lowest_left_child_impurity = current_left_impurity
                lowest_right_child_impurity = current_right_impurity

    return best_feature_id, best_threshold, lowest_left_child_impurity, lowest_right_child_impurity

class MyDecisionTreeNode:
    def __init__(
        self,
        meta: 'MyDecisionTreeClassifier',
        depth: int,
        node_type: NodeType = NodeType.REGULAR,
        predicted_class: tp.Optional[int] = None,
        left_subtree: tp.Optional['MyDecisionTreeNode'] = None,
        right_subtree: tp.Optional['MyDecisionTreeNode'] = None,
        feature_id: tp.Optional[int] = None,
        threshold: tp.Optional[float] = None,
        impurity: float = np.inf
    ):
        self._node_type = node_type
        self._meta = meta
        self._depth = depth
        self._predicted_class = predicted_class
        self._class_proba = None
        self._left_subtree = left_subtree
        self._right_subtree = right_subtree
        self._feature_id = feature_id
        self._threshold = threshold
        self._impurity = impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tp.Tuple[int, float, float, float]:
        lowest_impurity = np.inf
        best_feature_id, best_threshold = None, None
        lowest_left_impurity, lowest_right_impurity = None, None
        features = self._meta.rng.permutation(X.shape[1])

        for feature in features:
            current_feature_values = X[:, feature]
            thresholds = np.unique(current_feature_values)
            
            for threshold in thresholds:
                left_idx, right_idx = create_split(current_feature_values, threshold)
                if np.any(left_idx) and np.any(right_idx):
                    weighted_imp, left_imp, right_imp = weighted_impurity(y[left_idx], y[right_idx])
                    
                    if weighted_imp < lowest_impurity:
                        lowest_impurity = weighted_imp
                        best_feature_id = feature
                        best_threshold = threshold
                        lowest_left_impurity = left_imp
                        lowest_right_impurity = right_imp

        return best_feature_id, best_threshold, lowest_left_impurity, lowest_right_impurity

    def fit(self, X: np.ndarray, y: np.ndarray):
        if (self._depth >= self._meta.max_depth or y.size < self._meta.min_samples_split or gini(y) == 0):
            self._node_type = NodeType.TERMINAL
            self._predicted_class = mode(y, keepdims=True)[0][0]
            class_counts = np.bincount(y, minlength=self._meta._n_classes)
            self._class_proba = class_counts / y.size
            return self
        
        self._feature_id, self._threshold, _, _ = self._best_split(X, y)
        if self._feature_id is None:
            self._node_type = NodeType.TERMINAL
            self._predicted_class = mode(y, keepdims=True)[0][0]
            class_counts = np.bincount(y, minlength=self._meta._n_classes)
            self._class_proba = class_counts / y.size
            return self

        left_idx, right_idx = create_split(X[:, self._feature_id], self._threshold)
        self._left_subtree = MyDecisionTreeNode(
            meta=self._meta,
            depth=self._depth + 1
        ).fit(X[left_idx], y[left_idx])

        self._right_subtree = MyDecisionTreeNode(
            meta=self._meta,
            depth=self._depth + 1
        ).fit(X[right_idx], y[right_idx])

        return self

    def predict(self, x: np.ndarray) -> int:
        if self._node_type == NodeType.TERMINAL:
            return self._predicted_class
        if x[self._feature_id] <= self._threshold:
            return self._left_subtree.predict(x)
        return self._right_subtree.predict(x)


class MyDecisionTreeClassifier:
    def __init__(self, max_depth: tp.Optional[int] = None, min_samples_split: tp.Optional[int] = 2, seed: int = 0):
        self.root = None
        self._is_trained = False
        self.max_depth = max_depth or np.inf
        self.min_samples_split = min_samples_split or 2
        self.rng = np.random.default_rng(seed)
        self._n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_classes = np.unique(y).shape[0]
        self.root = MyDecisionTreeNode(meta=self, depth=1)
        self.root.fit(X, y)
        self._is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be fitted before calling predict.")
        return np.array([self.root.predict(x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        return np.array([self.root._class_proba for x in X])

import numpy as np
from typing import Optional, Tuple

class MyRandomForestClassifier:
    """
    Data-diverse ensemble of tree classifiers
    """
    big_number = 1 << 32

    def __init__(
            self,
            n_estimators: int,
            max_depth: Optional[int] = None,
            min_samples_split: Optional[int] = 2,
            seed: int = 0
    ):
        self._n_classes = 0
        self._is_trained = False
        self.rng = np.random.default_rng(seed)
        self.estimators = [
            MyDecisionTreeClassifier(max_depth, min_samples_split, seed=seed) 
            for seed in self.rng.choice(
                max(MyRandomForestClassifier.big_number, n_estimators), 
                size=(n_estimators,), 
                replace=False)
        ]

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        indices = self.rng.choice(len(X), size=len(X), replace=True)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_classes = np.unique(y).shape[0]
        for estimator in self.estimators:
            X_sample, y_sample = self._bootstrap_sample(X, y)
            estimator.fit(X_sample, y_sample)
        self._is_trained = True
        return self

    def predict_proba(self, X: np.ndarray):
        probas = np.zeros((X.shape[0], self._n_classes))
        for estimator in self.estimators:
            probas += estimator.predict_proba(X)
        probas /= len(self.estimators)
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

class Logger:
    """Logger performs data management and stores scores and other relevant information"""

    def __init__(self, logs_path: tp.Union[str, os.PathLike]):
        self.path = pathlib.Path(logs_path)

        records = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith('.json'):
                    uuid = os.path.splitext(file)[0]
                    with open(os.path.join(root, file), 'r') as f:
                        try:
                            logged_data = json.load(f)
                            records.append(
                                {
                                    'id': uuid,
                                    **logged_data
                                }
                            )
                        except json.JSONDecodeError:
                            pass
        if records:
            self.leaderboard = pd.DataFrame.from_records(records, index='id')
        else:
            self.leaderboard = pd.DataFrame(index=pd.Index([], name='id'))

        self._current_run = None

    class Run:
        """Run incapsulates information for a particular entry of logged material. Each run is solitary experiment"""

        def __init__(self, name, storage, path):
            self.name = name
            self._storage = storage
            self._path = path
            self._storage.append(pd.Series(name=name))

        def log(self, key, value):
            self._storage.loc[self.name, key] = value

        def log_values(self, log_values: tp.Dict[str, tp.Any]):
            for key, value in log_values.items():
                self.log(key, value)

        def save_logs(self):
            with open(self._path / f'{self.name}.json', 'w+') as f:
                json.dump(self._storage.loc[self.name].to_dict(), f)

        def log_artifact(self, fname: str, writer: tp.Callable):
            with open(self._path / fname, 'wb+') as f:
                writer(f)

    @contextlib.contextmanager
    def run(self, name: tp.Optional[str] = None):
        if name is None:
            name = str(uuid.uuid4())
        elif name in self.leaderboard.index:
            raise NameError("Run with given name already exists, name should be unique")
        else:
            name = name.replace(' ', '_')
        self._current_run = Logger.Run(name, self.leaderboard, self.path / name)
        os.makedirs(self.path / name, exist_ok=True)
        try:
            yield self._current_run
        finally:
            self._current_run.save_logs()


def load_predictions_dataframe(filename, column_prefix, index):
    with open(filename, 'rb') as file:
        data = np.load(file)
        dataframe = pd.DataFrame(data, columns=[f'{column_prefix}_{i}' for i in range(data.shape[1])],
                                 index=index)
        return dataframe


class ExperimentHandler:
    """This class perfoms experiments with given model, measures metrics and logs everything for thorough comparison"""
    stacking_prediction_filename = 'cv_stacking_prediction.npy'
    test_stacking_prediction_filename = 'test_stacking_prediction.npy'

    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            cv_iterable: tp.Union[sklearn.model_selection.KFold, tp.Iterable],
            logger: Logger,
            metrics: tp.Dict[str, tp.Union[tp.Callable, str]],
            n_jobs=-1
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._cv_iterable = cv_iterable
        self.logger = logger
        self._metrics = metrics
        self._n_jobs = n_jobs

    def score_test(self, estimator, metrics, run, test_data=None):
        """
        Computes scores for test data and logs them to given run
        :param estimator: fitted estimator
        :param metrics: metrics to compute
        :param run: run to log into
        :param test_data: optional argument if one wants to pass augmented test dataset
        :return: None
        """
        if test_data is None:
            test_data = self.X_test
        test_scores = _score(estimator, test_data, self.y_test, metrics)
        run.log_values({key + '_test': value for key, value in test_scores.items()})

    def score_cv(self, estimator, metrics, run):
        """
        computes scores on cross-validation
        :param estimator: estimator to fit
        :param metrics: metrics to compute
        :param run: run to log to
        :return: None
        """
        raise NotImplementedError("COMPLETE THIS FUNCTION")
        cross_val_results = ...  # compute crossval scores (easier done using corresponding sklearn method)
        for key, value in cross_val_results.items():
            if key.startswith('test_'):
                metric_name = key.split('_', maxsplit=1)[1]
                mean_score = np.mean(value)
                std_score = np.std(value)
                run.log_values(
                    {
                        metric_name + '_mean': mean_score,
                        metric_name + '_std': std_score
                    }
                )

    def generate_stacking_predictions(self, estimator, run):
        """
        generates predictions over cross-validation folds, then saves them as artifacts
        returns fitted estimator for convinience and les train overhead
        :param estimator: estimator to use
        :param run: run to log to
        :return: estimator fitted on train, stacking cross-val predictions, stacking test predictions
        """
        raise NotImplementedError("COMPLETE THIS FUNCTION")
        if hasattr(estimator, "predict_proba"):  # choose the most informative method for stacking predictions
            method = "predict_proba"
        elif hasattr(estimator, "decision_function"):
            method = "decision_function"
        else:
            method = "predict"
        cross_val_stacking_prediction = ...  # generate crossval predictions for stacking using most informative method
        run.log_artifact(ExperimentHandler.stacking_prediction_filename,
                         lambda file: np.save(file, cross_val_stacking_prediction))
        estimator.fit(self.X_train, self.y_train)
        test_stacking_prediction = getattr(estimator, method)(self.X_test)
        run.log_artifact(ExperimentHandler.test_stacking_prediction_filename,
                         lambda file: np.save(file, test_stacking_prediction))
        return estimator, cross_val_stacking_prediction, test_stacking_prediction

    def get_metrics(self, estimator):
        """
        get callable metrics with estimator validation
        (e.g. estimator has predict_proba necessary for likelihood computation, etc)
        """
        return _check_multimetric_scoring(estimator, self._metrics)

    def run(self, estimator: sklearn.base.BaseEstimator, name=None):
        """
        perform run for given estimator
        :param estimator: estimator to use
        :param name: name of run for convinience and consitent logging
        :return: leaderboard with conducted run
        """
        metrics = self.get_metrics(estimator)
        with self.logger.run(name=name) as run:
            # compute predictions over cross-validation
            self.score_cv(estimator, metrics, run)
            fitted_on_train, _, _ = self.generate_stacking_predictions(estimator, run)
            self.score_test(fitted_on_train, metrics, run, test_data=self.X_test)
            return self.logger.leaderboard.loc[[run.name]]

    def get_stacking_predictions(self, run_names):
        """
        :param run_names: run names for which to extract stacking predictions for averaging and stacking
        :return: dataframe with predictions indexed by run names
        """
        train_dataframes = []
        test_dataframes = []
        for run_name in run_names:
            train_filename = self.logger.path / run_name / ExperimentHandler.stacking_prediction_filename
            train_dataframes.append(load_predictions_dataframe(filename=train_filename, column_prefix=run_name,
                                                               index=self.X_train.index))
            test_filename = self.logger.path / run_name / ExperimentHandler.test_stacking_prediction_filename
            test_dataframes.append(load_predictions_dataframe(filename=test_filename, column_prefix=run_name,
                                                              index=self.X_test.index))

        return pd.concat(train_dataframes, axis=1), pd.concat(test_dataframes, axis=1)
