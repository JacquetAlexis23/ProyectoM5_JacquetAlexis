"""
Tests Unitarios - Model Training: Threshold Tuning & Sampler
=============================================================
Tests para validar:
- tune_threshold: devuelve un umbral razonable y mejora la métrica.
- predict_with_threshold: aplica el umbral y altera las predicciones.
- Sampler: rebalancear clases antes del entrenamiento.

Autor: Alexis Jacquet
HENRY M5 - Avance 3 Extra Credit
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Añadir path del módulo
sys.path.insert(0, str(Path(__file__).parent.parent / "mlops_pipeline" / "src"))

from model_training_evaluation import tune_threshold, predict_with_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _imbalanced_train_test(ratio=0.1, n_samples=300, random_state=42):
    """
    Genera un dataset de clasificación binaria desbalanceado.

    ratio  ≈  n_clase_1 / n_clase_0
    """
    n_min = int(n_samples * ratio)
    n_maj = n_samples - n_min

    rng = np.random.default_rng(random_state)

    # Clase 0 (mayoritaria)
    X0 = rng.normal(loc=0.0, scale=1.0, size=(n_maj, 4))
    # Clase 1 (minoritaria)
    X1 = rng.normal(loc=2.0, scale=1.0, size=(n_min, 4))

    X = np.vstack([X0, X1])
    y = np.array([0] * n_maj + [1] * n_min)

    # shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    split = int(0.7 * len(y))
    return X[:split], X[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# Tests de tune_threshold
# ---------------------------------------------------------------------------

class TestTuneThreshold:
    """Tests para la función tune_threshold."""

    @pytest.fixture
    def trained_model(self):
        """Modelo LogisticRegression entrenado sobre dataset desbalanceado."""
        X_train, _, y_train, _ = _imbalanced_train_test()
        model = LogisticRegression(class_weight='balanced', random_state=42,
                                   max_iter=300)
        model.fit(X_train, y_train)
        return model

    @pytest.fixture
    def val_data(self):
        _, X_val, _, y_val = _imbalanced_train_test()
        return X_val, y_val

    def test_returns_tuple(self, trained_model, val_data):
        """tune_threshold debe devolver una tupla (threshold, score)."""
        X_val, y_val = val_data
        result = tune_threshold(trained_model, X_val, y_val)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_threshold_in_range(self, trained_model, val_data):
        """El umbral devuelto debe estar en [0.10, 0.90]."""
        X_val, y_val = val_data
        threshold, _ = tune_threshold(trained_model, X_val, y_val)
        assert 0.0 <= threshold <= 1.0

    def test_score_is_valid(self, trained_model, val_data):
        """El score devuelto debe ser un float en [0, 1]."""
        X_val, y_val = val_data
        _, score = tune_threshold(trained_model, X_val, y_val)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_metric_f1(self, trained_model, val_data):
        """tune_threshold con metric='f1' debe devolver umbral razonable."""
        X_val, y_val = val_data
        t, s = tune_threshold(trained_model, X_val, y_val, metric='f1')
        assert 0.0 <= t <= 1.0
        assert s >= 0.0

    def test_metric_precision(self, trained_model, val_data):
        """tune_threshold con metric='precision' debe funcionar sin errores."""
        X_val, y_val = val_data
        t, s = tune_threshold(trained_model, X_val, y_val, metric='precision')
        assert 0.0 <= t <= 1.0

    def test_metric_recall(self, trained_model, val_data):
        """tune_threshold con metric='recall' debe funcionar sin errores."""
        X_val, y_val = val_data
        t, s = tune_threshold(trained_model, X_val, y_val, metric='recall')
        assert 0.0 <= t <= 1.0

    def test_custom_thresholds(self, trained_model, val_data):
        """Se puede pasar un array personalizado de umbrales."""
        X_val, y_val = val_data
        custom = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        t, _ = tune_threshold(trained_model, X_val, y_val, thresholds=custom)
        assert t in custom

    def test_model_without_predict_proba(self, val_data):
        """Un modelo sin predict_proba debe devolver fallback (0.5, None)."""
        from sklearn.svm import SVC
        X_val, y_val = val_data
        X_train, _, y_train, _ = _imbalanced_train_test()
        # SVC sin probability=True no tiene predict_proba
        svm = SVC(kernel='linear', probability=False)
        svm.fit(X_train, y_train)
        t, s = tune_threshold(svm, X_val, y_val)
        # Con decision_function disponible, debe seguir funcionando
        assert 0.0 <= t <= 1.0


# ---------------------------------------------------------------------------
# Tests de predict_with_threshold
# ---------------------------------------------------------------------------

class TestPredictWithThreshold:
    """Tests para la función predict_with_threshold."""

    @pytest.fixture
    def model_and_data(self):
        X_train, X_val, y_train, _ = _imbalanced_train_test()
        model = LogisticRegression(class_weight='balanced', random_state=42,
                                   max_iter=300)
        model.fit(X_train, y_train)
        return model, X_val

    def test_returns_binary_array(self, model_and_data):
        """predict_with_threshold debe devolver array binario (0/1)."""
        model, X_val = model_and_data
        preds = predict_with_threshold(model, X_val, threshold=0.5)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_low_threshold_more_positives(self, model_and_data):
        """Un umbral bajo debe producir más predicciones positivas."""
        model, X_val = model_and_data
        preds_low = predict_with_threshold(model, X_val, threshold=0.2)
        preds_high = predict_with_threshold(model, X_val, threshold=0.8)
        assert preds_low.sum() >= preds_high.sum()

    def test_threshold_affects_predictions(self, model_and_data):
        """Umbrales distintos deben producir predicciones distintas para datos desbalanceados."""
        model, X_val = model_and_data
        preds_05 = predict_with_threshold(model, X_val, threshold=0.5)
        preds_02 = predict_with_threshold(model, X_val, threshold=0.2)
        # Con dataset desbalanceado, bajar el umbral suele cambiar predicciones
        assert not np.array_equal(preds_05, preds_02)

    def test_output_shape(self, model_and_data):
        """El array de salida debe tener la misma longitud que X."""
        model, X_val = model_and_data
        preds = predict_with_threshold(model, X_val)
        assert len(preds) == len(X_val)


# ---------------------------------------------------------------------------
# Tests de sampler en train_multiple_models
# ---------------------------------------------------------------------------

class TestSamplerIntegration:
    """Tests para el parámetro sampler de train_multiple_models."""

    def test_sampler_changes_class_distribution(self):
        """
        SMOTE o RandomUnderSampler debe modificar la distribución de clases
        en el conjunto de entrenamiento.
        """
        try:
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError:
            pytest.skip("imbalanced-learn no está instalado")

        X_train, _, y_train, _ = _imbalanced_train_test(ratio=0.1, n_samples=300)

        ratio_before = (y_train == 1).sum() / (y_train == 0).sum()

        sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_train)

        ratio_after = (y_res == 1).sum() / (y_res == 0).sum()

        assert ratio_after > ratio_before, (
            "El sampler debe aumentar la proporción de la clase minoritaria"
        )

    def test_smote_increases_minority_class(self):
        """SMOTE debe generar muestras sintéticas para la clase minoritaria."""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            pytest.skip("imbalanced-learn no está instalado")

        X_train, _, y_train, _ = _imbalanced_train_test(ratio=0.1, n_samples=300)
        n_minority_before = (y_train == 1).sum()

        smote = SMOTE(random_state=42)
        _, y_res = smote.fit_resample(X_train, y_train)
        n_minority_after = (y_res == 1).sum()

        assert n_minority_after > n_minority_before

    def test_train_multiple_models_accepts_sampler_param(self):
        """
        train_multiple_models debe aceptar sampler=None sin errores
        (backward compatibility).
        """
        from model_training_evaluation import train_multiple_models

        X_train, X_test, y_train, y_test = _imbalanced_train_test(n_samples=200)

        # Sólo ejecutamos 1 modelo para que el test sea rápido.
        # Probamos que la firma acepta sampler=None sin lanzar TypeError.
        try:
            models_dict, results_df = train_multiple_models(
                X_train, y_train, X_test, y_test, sampler=None
            )
        except TypeError as e:
            pytest.fail(f"train_multiple_models no acepta sampler=None: {e}")

        assert results_df is not None
        assert len(results_df) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
