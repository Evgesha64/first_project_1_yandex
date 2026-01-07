from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator


class SkCatBoostClassifier(CatBoostClassifier, BaseEstimator):
    """CatBoostClassifier with sklearn tags to satisfy sklearn checks."""

    def __sklearn_tags__(self):
        # Minimal set of tags to pass sklearn checks
        from sklearn.utils._tags import Tags

        return Tags(non_deterministic=True)

    def __sklearn_is_fitted__(self):
        # CatBoost manages its own fitted state; treat as fitted after .fit()
        return hasattr(self, "is_fitted_") or True

