import numpy
import numpy as np
import sklearn.linear_model
from scipy.stats import t, norm
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class LogReg(LogisticRegression):

    def __init__(
        self,
        penalty="l2",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
    ):

        super(LogReg, self).__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
        )
        self.ci = None

    def compute_ci(self, X: np.ndarray, confidence: float = 0.95):
        """

        :param X:
        :param confidence:
        :return:
        """

        check_is_fitted(self)
        n_samples, n_features = X.shape

        # Predict the probabilities of the positive class
        p = self.predict_proba(X)[:, 1]
        if self.fit_intercept:
            intercept = np.ones([n_samples, 1])
            X = np.hstack([intercept, X])
            coefficients = np.hstack([self.intercept_, self.coef_.flatten()])
        else:
            coefficients = self.coef_.flatten()

        # Calculate the covariance matrix (Hessian inverse)
        # Variance for each coefficient: (X.T @ W @ X)^(-1), where W is the diagonal matrix of p*(1-p)
        W = np.diag(p * (1 - p))
        XtW = np.dot(X.T, W)
        cov_matrix = np.linalg.inv(np.dot(XtW, X))

        # Get the standard errors (square root of the diagonal elements of the covariance matrix)
        standard_errors = np.sqrt(np.diag(cov_matrix))

        # Get the z-score for the given confidence level
        z = norm.ppf(1 - (1 - confidence) / 2)

        # Calculate the confidence intervals for each coefficient
        ci_lower = coefficients - z * standard_errors
        ci_upper = coefficients + z * standard_errors

        self.ci = np.vstack((ci_lower, ci_upper)).T

        return self


def logistic_regression_ci(
    model: sklearn.linear_model, X: np.ndarray, confidence: float = 0.95
) -> np.ndarray:
    """
    Calculate the confidence intervals for the coefficients of a fitted logistic regression model.

    :param model: A fitted sklearn LogisticRegression model.
    :param X: Feature matrix (same that was used to fit the model).
    :param confidence: Confidence level for the intervals (default is 0.95).
    :return: A 2D array of confidence intervals for each coefficient.
    """

    check_is_fitted(model)

    # Number of samples and features
    n_samples, n_features = X.shape

    # Predict the probabilities of the positive class
    p = model.predict_proba(X)[:, 1]

    if model.fit_intercept:
        intercept = np.ones([n_samples, 1])
        X = np.hstack([intercept, X])
        coefficients = np.hstack([model.intercept_, model.coef_.flatten()])
    else:
        coefficients = model.coef_.flatten()

    # Calculate the covariance matrix (Hessian inverse)
    # Variance for each coefficient: (X.T @ W @ X)^(-1), where W is the diagonal matrix of p*(1-p)
    W = np.diag(p * (1 - p))
    XtW = np.dot(X.T, W)
    cov_matrix = np.linalg.inv(np.dot(XtW, X))

    # Get the standard errors (square root of the diagonal elements of the covariance matrix)
    standard_errors = np.sqrt(np.diag(cov_matrix))

    # Get the z-score for the given confidence level
    z = norm.ppf(1 - (1 - confidence) / 2)

    # Calculate the confidence intervals for each coefficient
    ci_lower = coefficients - z * standard_errors
    ci_upper = coefficients + z * standard_errors

    return np.vstack((ci_lower, ci_upper)).T
