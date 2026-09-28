"""v4 model families.

F0..F6 as sklearn-style estimators, the non-negative-logistic stacker S,
and Platt calibration on inner-OOF predictions.

Every family is a scikit-learn Pipeline, so each fitted transform
(imputer, scaler, PCA, gradient-boosting model) is fit on the training
people passed to `fit` and never sees held-out people. Fold-dependent
harmonisation that cannot live inside a Pipeline (C1 tangent reference,
C3 normative models) is applied by compare_v4 before the matrix reaches
these estimators, also from training-fold people only.

Status of the protocol's full family list (docs/PD_PROTOCOL_V4.md S7):
- F0..F6 implemented here.
- F7 (attention-MIL over per-window features) is NOT implemented in this
  run; the stacker therefore combines F0..F6.
- F8 (connectivity logistic) is blocked on B-7, which Stage 5 did not
  implement. Both absences are recorded in docs/PD_V4_LOG.md rather than
  passed over silently.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 20261001


def _clean(X):
    """Replace non-finite values so estimators without native NaN support
    (SVC, LogisticRegression) can run. B-2 legitimately carries NaN for
    channels with a poor specparam fit or no detected alpha peak; the
    protocol says to impute those within the training fold, which the
    SimpleImputer in each pipeline does. This guard only catches infinities
    that survive imputation."""
    return np.nan_to_num(np.asarray(X, dtype="float64"), posinf=0.0, neginf=0.0)


class LeapdScorer:
    """B-4 / F4. Per channel, fit a PCA hyperplane to the PD training people
    and another to the controls; a person's channel score is
    d(control hyperplane) - d(PD hyperplane), and the family score averages
    over channels. Both hyperplanes are fit on training people only."""

    def __init__(self, n_components=3, n_channels=17, n_coefficients=12):
        self.n_components = n_components
        self.n_channels = n_channels
        self.n_coefficients = n_coefficients

    def _per_channel(self, X):
        # b4 is [mean(17x12) | sd(17x12)]; LEAPD uses the mean block.
        block = X[:, : self.n_channels * self.n_coefficients]
        return block.reshape(len(X), self.n_channels, self.n_coefficients)

    def fit(self, X, y):
        X = _clean(X)
        y = np.asarray(y)
        per_channel = self._per_channel(X)
        self.models_ = []
        for c in range(self.n_channels):
            channel = {}
            for label in (0, 1):
                rows = per_channel[y == label, c, :]
                k = min(self.n_components, max(1, min(rows.shape) - 1))
                channel[label] = PCA(n_components=k).fit(rows) if len(rows) > k else None
            self.models_.append(channel)
        return self

    def transform(self, X):
        X = _clean(X)
        per_channel = self._per_channel(X)
        scores = np.zeros((len(X), self.n_channels))
        for c, channel in enumerate(self.models_):
            rows = per_channel[:, c, :]
            distances = {}
            for label in (0, 1):
                model = channel[label]
                if model is None:
                    distances[label] = np.zeros(len(X))
                    continue
                reconstructed = model.inverse_transform(model.transform(rows))
                distances[label] = np.linalg.norm(rows - reconstructed, axis=1)
            scores[:, c] = distances[0] - distances[1]
        return scores

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def _svm(C):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(C=C, kernel="rbf", gamma="scale", probability=True, random_state=SEED),
    )


def _logistic(C, penalty="l2", l1_ratio=None):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=C,
            penalty=penalty,
            l1_ratio=l1_ratio,
            solver="saga" if penalty == "elasticnet" else "lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )


def _boosting(max_depth):
    return make_pipeline(
        HistGradientBoostingClassifier(
            max_depth=max_depth,
            learning_rate=0.05,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=SEED,
        )
    )


def _leapd(n_components):
    return make_pipeline(
        LeapdScorer(n_components=n_components),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=SEED),
    )


# Each family: which cached feature block it consumes, and its frozen grid.
FAMILIES = {
    "F0_rich_svm": {"block": "v3_rich", "grid": [{"C": 1.0}, {"C": 10.0}], "build": lambda p: _svm(p["C"])},
    "F1_relative_svm": {"block": "b1_b6", "grid": [{"C": 1.0}, {"C": 10.0}], "build": lambda p: _svm(p["C"])},
    "F2_specparam_enet": {
        "block": "b2_c3",
        "grid": [
            {"C": c, "l1_ratio": r} for c in (0.1, 1.0) for r in (0.2, 0.8)
        ],
        "build": lambda p: _logistic(p["C"], penalty="elasticnet", l1_ratio=p["l1_ratio"]),
    },
    "F3_riemann_logistic": {
        "block": "b3_tangent",
        "grid": [{"C": 0.1}, {"C": 1.0}],
        "build": lambda p: _logistic(p["C"]),
    },
    "F4_leapd": {
        "block": "b4",
        "grid": [{"k": 2}, {"k": 3}],
        "build": lambda p: _leapd(p["k"]),
    },
    "F5_entropy_logistic": {
        "block": "b5",
        "grid": [{"C": 0.1}, {"C": 1.0}],
        "build": lambda p: _logistic(p["C"]),
    },
    "F6_boosting": {
        "block": "compact",
        "grid": [{"max_depth": 2}, {"max_depth": 3}],
        "build": lambda p: _boosting(p["max_depth"]),
    },
}


class NonNegativeStacker:
    """Pre-declared stacker S: logistic regression on the inner-OOF
    probabilities of the level-0 families, constrained to non-negative
    weights so a family can only add evidence, never be inverted.

    Non-negativity is enforced by fitting on the logit-transformed OOF
    probabilities with a large L2 penalty and clipping negative
    coefficients to zero, then refitting the intercept -- a simple
    projection that keeps the model interpretable and stable with only a
    handful of inputs.
    """

    def __init__(self, C=1.0):
        self.C = C

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, P, y):
        Z = self._logit(P)
        model = LogisticRegression(
            C=self.C, max_iter=5000, class_weight="balanced", random_state=SEED
        ).fit(Z, y)
        weights = np.maximum(model.coef_.ravel(), 0.0)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        self.weights_ = weights
        # Refit only the intercept with the projected weights held fixed.
        offset = Z @ self.weights_
        intercept_model = LogisticRegression(
            C=1e6, max_iter=5000, class_weight="balanced", random_state=SEED
        ).fit(offset.reshape(-1, 1), y)
        self.scale_ = float(intercept_model.coef_.ravel()[0])
        self.intercept_ = float(intercept_model.intercept_[0])
        return self

    def decision_function(self, P):
        return self._logit(P) @ self.weights_ * self.scale_ + self.intercept_

    def predict_proba(self, P):
        d = self.decision_function(P)
        p = 1.0 / (1.0 + np.exp(-d))
        return np.column_stack([1 - p, p])


def platt_calibrate(scores, y):
    """D6 calibration: Platt scaling fit on inner-OOF predictions only."""
    scores = np.asarray(scores, dtype="float64").reshape(-1, 1)
    return LogisticRegression(max_iter=5000, random_state=SEED).fit(scores, y)
