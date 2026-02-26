"""
Tests Unitarios - Feature Engineering
======================================
Tests para validar pipeline de feature engineering

Autor: Alexis Jacquet
HENRY M5 - Avance 3 Extra Credit
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Añadir path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlops_pipeline" / "src"))

from ft_engineering import DateFeatureEngineer, FinancialFeatureEngineer, prepare_features, LEAKAGE_COLUMNS


# ---------------------------------------------------------------------------
# Fixture de DataFrame completo para tests de prepare_features
# ---------------------------------------------------------------------------

def _make_full_df(n=10, seed=0):
    """Crea un DataFrame mínimo con todas las columnas que necesita prepare_features."""
    rng = np.random.default_rng(seed)
    dates = ['2024-06-15', '2024-07-20', '2024-08-10',
             '2024-09-05', '2024-10-01', '2024-11-12',
             '2024-12-01', '2025-01-15', '2025-02-20', '2025-03-10']
    return pd.DataFrame({
        'fecha_prestamo':                [dates[i % len(dates)] for i in range(n)],
        'tipo_credito':                  rng.integers(1, 4, n).tolist(),
        'capital_prestado':              rng.uniform(1e6, 10e6, n).tolist(),
        'plazo_meses':                   rng.integers(12, 60, n).tolist(),
        'edad_cliente':                  rng.integers(25, 60, n).tolist(),
        'tipo_laboral':                  ['Empleado'] * n,
        'salario_cliente':               rng.uniform(1.5e6, 5e6, n).tolist(),
        'total_otros_prestamos':         rng.uniform(0, 1e6, n).tolist(),
        'cuota_pactada':                 rng.uniform(1e5, 4e5, n).tolist(),
        'puntaje':                       rng.uniform(500, 900, n).tolist(),
        'puntaje_datacredito':           rng.uniform(450, 850, n).tolist(),
        'cant_creditosvigentes':         rng.integers(1, 5, n).tolist(),
        'huella_consulta':               rng.integers(0, 15, n).tolist(),
        'saldo_mora':                    rng.uniform(0, 5e5, n).tolist(),
        'saldo_total':                   rng.uniform(1e6, 9e6, n).tolist(),
        'saldo_principal':               rng.uniform(1e6, 9e6, n).tolist(),
        'saldo_mora_codeudor':           rng.uniform(0, 1e5, n).tolist(),
        'creditos_sectorFinanciero':     rng.integers(0, 4, n).tolist(),
        'creditos_sectorCooperativo':    rng.integers(0, 3, n).tolist(),
        'creditos_sectorReal':           rng.integers(0, 3, n).tolist(),
        'promedio_ingresos_datacredito': rng.uniform(1e6, 4e6, n).tolist(),
        'tendencia_ingresos':            ['Estable'] * n,
        'Pago_atiempo':                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1][:n],
    })


class TestDateFeatureEngineer:
    """Tests para DateFeatureEngineer"""
    
    @pytest.fixture
    def sample_data(self):
        """Datos de muestra con fechas"""
        return pd.DataFrame({
            'fecha_prestamo': ['2024-06-15', '2024-07-20', '2024-08-10'],
            'tipo_credito': [1, 2, 1],
            'capital_prestado': [5000000, 8000000, 3000000]
        })
    
    def test_date_feature_creation(self, sample_data):
        """Test creación de features de fecha"""
        engineer = DateFeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        assert 'mes_prestamo' in result.columns
        assert 'dia_semana' in result.columns
        assert 'trimestre' in result.columns
        assert 'dias_desde_epoca' in result.columns
        assert 'fecha_prestamo' not in result.columns  # Debe ser removida
    
    def test_mes_prestamo_range(self, sample_data):
        """Test rango válido de mes_prestamo"""
        engineer = DateFeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        assert result['mes_prestamo'].min() >= 1
        assert result['mes_prestamo'].max() <= 12
    
    def test_dia_semana_range(self, sample_data):
        """Test rango válido de dia_semana"""
        engineer = DateFeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        assert result['dia_semana'].min() >= 0
        assert result['dia_semana'].max() <= 6
    
    def test_trimestre_range(self, sample_data):
        """Test rango válido de trimestre"""
        engineer = DateFeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        assert result['trimestre'].min() >= 1
        assert result['trimestre'].max() <= 4


class TestFinancialFeatureEngineer:
    """Tests para FinancialFeatureEngineer"""
    
    @pytest.fixture
    def sample_data(self):
        """Datos de muestra financieros"""
        return pd.DataFrame({
            'salario_cliente': [3000000, 4500000, 2000000],
            'cuota_pactada': [180000, 250000, 140000],
            'saldo_total': [4500000, 7500000, 2800000],
            'capital_prestado': [5000000, 8000000, 3000000],
            'saldo_mora': [0, 150000, 0],
            'saldo_mora_codeudor': [0, 50000, 0],
            'cant_creditosvigentes': [2, 3, 1],
            'creditos_sectorFinanciero': [1, 2, 1],
            'creditos_sectorCooperativo': [1, 0, 0],
            'creditos_sectorReal': [0, 1, 0],
            'puntaje': [750, 680, 800],
            'puntaje_datacredito': [720, 650, 780],
            'total_otros_prestamos': [500000, 1200000, 200000]
        })
    
    def test_financial_feature_creation(self, sample_data):
        """Test creación de features financieras"""
        engineer = FinancialFeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        assert 'ratio_cuota_salario' in result.columns
        assert 'ratio_deuda_ingreso' in result.columns
        assert 'ratio_capital_salario' in result.columns
        assert 'tiene_mora' in result.columns
        assert 'tiene_codeudor_mora' in result.columns
    
    def test_ratio_calculations(self, sample_data):
        """Test cálculos de ratios"""
        engineer = FinancialFeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        # Verificar que los ratios son razonables
        assert result['ratio_cuota_salario'].min() >= 0
        assert result['ratio_deuda_ingreso'].min() >= 0
        assert result['ratio_capital_salario'].min() >= 0
    
    def test_mora_indicators(self, sample_data):
        """Test indicadores de mora"""
        engineer = FinancialFeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        # Verificar que son binarios (0 o 1)
        assert set(result['tiene_mora'].unique()).issubset({0, 1})
        assert set(result['tiene_codeudor_mora'].unique()).issubset({0, 1})
    
    def test_no_nan_values(self, sample_data):
        """Test que no hay valores NaN en el resultado"""
        engineer = FinancialFeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        # No debe haber NaN después de la transformación
        assert not result.isnull().any().any()
    
    def test_no_inf_values(self, sample_data):
        """Test que no hay valores infinitos"""
        engineer = FinancialFeatureEngineer()
        result = engineer.fit_transform(sample_data)
        
        # No debe haber infinitos
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()


@pytest.mark.unit
class TestFeatureEngineeringUtils:
    """Tests de utilidades de feature engineering"""
    
    def test_division_by_zero_handling(self):
        """Test manejo de división por cero usando np.divide seguro"""
        salario = np.array([0.0, 3000000.0])
        cuota = np.array([180000.0, 180000.0])

        # np.divide con where evita la evaluación cuando salario == 0
        ratio = np.divide(
            cuota, salario,
            where=(salario > 0),
            out=np.zeros_like(cuota, dtype=float)
        )
        ratio = np.where(salario > 0, ratio, 0.0)

        assert ratio[0] == 0.0          # salario=0  -> ratio=0, sin ZeroDivisionError
        assert ratio[1] == pytest.approx(0.06, rel=1e-3)  # 180000/3000000
    
    def test_negative_values_handling(self):
        """Test manejo de valores negativos"""
        mora = 0
        
        tiene_mora = int(mora > 0)
        
        assert tiene_mora == 0


# ---------------------------------------------------------------------------
# Tests de eliminación de leakage (drop_leakage)
# ---------------------------------------------------------------------------

class TestDropLeakage:
    """Tests para el parámetro drop_leakage de prepare_features."""

    def test_leakage_columns_constant_defined(self):
        """LEAKAGE_COLUMNS debe estar definida y contener 'puntaje'."""
        assert isinstance(LEAKAGE_COLUMNS, list)
        assert 'puntaje' in LEAKAGE_COLUMNS

    def test_puntaje_present_by_default(self):
        """Sin drop_leakage la columna 'puntaje' aparece en las features."""
        df = _make_full_df()
        _, _, feature_names, _, _, _ = prepare_features(df, drop_leakage=False)
        assert 'puntaje' in list(feature_names), (
            "'puntaje' debe estar en feature_names cuando drop_leakage=False"
        )

    def test_puntaje_absent_with_drop_leakage(self):
        """Con drop_leakage=True la columna 'puntaje' no aparece en las features."""
        df = _make_full_df()
        _, _, feature_names, _, _, _ = prepare_features(df, drop_leakage=True)
        assert 'puntaje' not in list(feature_names), (
            "'puntaje' NO debe estar en feature_names cuando drop_leakage=True"
        )

    def test_drop_leakage_reduces_feature_count(self):
        """drop_leakage=True debe producir menos features que drop_leakage=False."""
        df = _make_full_df()
        _, _, fn_full, _, _, _ = prepare_features(df, drop_leakage=False)
        _, _, fn_clean, _, _, _ = prepare_features(df, drop_leakage=True)
        assert len(fn_clean) < len(fn_full), (
            "El número de features debe reducirse al eliminar columnas de leakage"
        )

    def test_ratio_puntajes_still_present_after_drop(self):
        """La feature derivada ratio_puntajes se conserva aunque puntaje (raw) se elimine."""
        df = _make_full_df()
        _, _, feature_names, _, _, _ = prepare_features(df, drop_leakage=True)
        assert 'ratio_puntajes' in list(feature_names), (
            "La feature derivada 'ratio_puntajes' debe conservarse incluso con drop_leakage=True"
        )

    def test_custom_leakage_cols(self):
        """leakage_cols personalizado elimina solo las columnas indicadas."""
        df = _make_full_df()
        _, _, fn_default, _, _, _ = prepare_features(df, drop_leakage=False)
        # Usamos una columna que existe en X pero que no estamos usando en prod.
        _, _, fn_custom, _, _, _ = prepare_features(
            df, drop_leakage=True, leakage_cols=['puntaje']
        )
        assert 'puntaje' not in list(fn_custom)
        assert len(fn_custom) < len(fn_default)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
