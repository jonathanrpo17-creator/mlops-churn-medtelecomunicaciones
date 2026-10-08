"""Modelos candidatos que se comparan en MLflow."""

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from proyecto_mlops.config import RANDOM_STATE


def get_candidates() -> dict[str, BaseEstimator]:
    """Nombre -> clasificador sin entrenar. El primero es el baseline."""
    return {
        "logreg_baseline": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "logreg_balanced": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }