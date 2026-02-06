"""
SVM classifier for binary classification evaluation

Uses StandardScaler preprocessing with SVC and RandomizedSearchCV for hyperparameter tuning.
"""

import numpy as np
import numpy.typing as npt
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(
    train_features: npt.NDArray[np.float32],
    train_labels: npt.NDArray[np.int_],
    n_iter: int = 100,
    n_jobs: int = 8,
    seed: int = 0,
) -> RandomizedSearchCV:
    """
    Train SVM with RandomizedSearchCV hyperparameter tuning

    Pipeline: StandardScaler -> SVC with hyperparameter search.

    Returns
    -------
    Fitted RandomizedSearchCV object.
    """

    svc = RandomizedSearchCV(
        make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf"),
        ),
        {
            "svc__C": scipy.stats.loguniform(1e-3, 1e1),
            "svc__kernel": ["rbf", "linear", "sigmoid", "poly"],
            "svc__gamma": scipy.stats.loguniform(1e-4, 1e-3),
        },
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=seed,
    )
    svc.fit(train_features, train_labels)

    return svc


def evaluate_svm(
    train_features: npt.NDArray[np.float32],
    train_labels: npt.NDArray[np.int_],
    test_features: npt.NDArray[np.float32],
    test_labels: npt.NDArray[np.int_],
    n_iter: int = 100,
    n_jobs: int = 8,
    seed: int = 0,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Evaluate using SVM with hyperparameter tuning

    Parameters
    ----------
    train_features
        Training features of shape (n_train, embedding_dim).
    train_labels
        Training labels of shape (n_train,).
    test_features
        Test features of shape (n_test, embedding_dim).
    test_labels
        Test labels of shape (n_test,).
    n_iter
        Number of RandomizedSearchCV iterations.
    n_jobs
        Number of parallel jobs for cross-validation.
    seed
        Random seed for reproducibility.

    Returns
    -------
    y_pred
        Predicted labels for test samples.
    y_true
        True labels for test samples (same as test_labels).
    """

    svc = train_svm(train_features, train_labels, n_iter=n_iter, n_jobs=n_jobs, seed=seed)
    y_pred = svc.predict(test_features)

    return (y_pred, test_labels)
