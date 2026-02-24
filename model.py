import numpy as np
from sklearn.linear_model import LinearRegression


def train_price_model(X, y):
    """
    Обучает модель линейной регрессии на наборе признаков X и целевой переменной y.
    X: np.ndarray, shape (n_samples, n_features)
    y: np.ndarray, shape (n_samples,)
    Возвращает обученную модель.
    """
    model = LinearRegression().fit(X, y)
    return model
