"""Explicit histogram estimation and Bayes decisions matching the course lectures."""
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def minimum_risk_decision(probabilities, loss=None):
    """Return action indices; loss[action, true_class]. Default is zero-one loss."""
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.ndim != 2 or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Expected normalized finite posterior rows.")
    c = probabilities.shape[1]
    loss = np.ones((c, c)) - np.eye(c) if loss is None else np.asarray(loss, dtype=float)
    if loss.shape != (c, c) or not np.isfinite(loss).all() or (loss < 0).any():
        raise ValueError("Loss must be a finite nonnegative class-by-class matrix.")
    return np.argmin(probabilities @ loss.T, axis=1)


class HistogramNaiveBayes(ClassifierMixin, BaseEstimator):
    """Shared feature bins, empirical class priors, symmetric count smoothing.

    Values beyond fitted support map to boundary bins. The likelihood describes
    this clipped representation; it does not estimate an unobserved tail density.
    Optional fixed edges support verification against the lecture's worked example.
    """

    def __init__(self, n_bins=10, alpha=1.0, bin_edges=None):
        self.n_bins = n_bins
        self.alpha = alpha
        self.bin_edges = bin_edges

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        if self.alpha < 0 or not np.isfinite(self.alpha):
            raise ValueError("alpha must be nonnegative and finite.")
        if not isinstance(self.n_bins, int) or self.n_bins < 1:
            raise ValueError("n_bins must be a positive integer.")
        self.n_features_in_ = X.shape[1]
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.class_count_ = np.bincount(encoded, minlength=len(self.classes_)).astype(float)
        self.class_prior_ = self.class_count_ / len(y)
        if self.bin_edges is not None and len(self.bin_edges) != self.n_features_in_:
            raise ValueError("One edge vector is required per feature.")
        self.bin_edges_, self.bin_count_, self.bin_probability_, self.bin_density_ = [], [], [], []
        for j in range(self.n_features_in_):
            edges = (np.histogram_bin_edges(X[:, j], bins=self.n_bins) if self.bin_edges is None
                     else np.asarray(self.bin_edges[j], dtype=float))
            if len(edges) < 2 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
                raise ValueError("Edges must be finite and strictly increasing.")
            if (X[:, j] < edges[0]).any() or (X[:, j] > edges[-1]).any():
                raise ValueError("Fixed edges must cover training observations.")
            bins = np.searchsorted(edges[1:-1], X[:, j], side="right")
            counts = np.array([np.bincount(bins[encoded == k], minlength=len(edges)-1)
                               for k in range(len(self.classes_))], dtype=float)
            probability = (counts + self.alpha) / (
                self.class_count_[:, None] + self.alpha * (len(edges)-1))
            self.bin_edges_.append(edges)
            self.bin_count_.append(counts)
            self.bin_probability_.append(probability)
            self.bin_density_.append(probability / np.diff(edges)[None, :])
        return self

    def predict_log_proba(self, X):
        check_is_fitted(self, "class_prior_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected feature count.")
        joint = np.tile(np.log(self.class_prior_), (len(X), 1))
        for j, edges in enumerate(self.bin_edges_):
            bins = np.searchsorted(edges[1:-1], X[:, j], side="right")
            with np.errstate(divide="ignore"):
                joint += np.log(self.bin_density_[j][:, bins]).T
        norm = logsumexp(joint, axis=1, keepdims=True)
        if not np.isfinite(norm).all():
            raise ValueError("All classes have zero evidence; positive smoothing is required.")
        return joint - norm

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[minimum_risk_decision(self.predict_proba(X))]
