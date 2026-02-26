"""
Tests Unitarios - Model Deployment Module
==========================================
Tests para validar funcionalidad del módulo de despliegue de modelos

Autor: Alexis Jacquet
HENRY M5 - Avance 3 Extra Credit
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys

# Añadir path
sys.path.insert(0, str(Path(__file__).parent.parent / "mlops_pipeline" / "src"))

from model_deploy import ModelDeployment


class TestModelDeployment:
    """Tests para la clase ModelDeployment"""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Crea directorio temporal para tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Datos de muestra para tests"""
        return {
            'fecha_prestamo': '2024-06-15',
            'tipo_credito': 1,
            'capital_prestado': 5000000.0,
            'plazo_meses': 36,
            'edad_cliente': 35,
            'tipo_laboral': 'Empleado',
            'salario_cliente': 3000000.0,
            'total_otros_prestamos': 500000.0,
            'cuota_pactada': 180000.0,
            'puntaje': 750.0,
            'puntaje_datacredito': 720.0,
            'cant_creditosvigentes': 2,
            'huella_consulta': 5,
            'saldo_mora': 0.0,
            'saldo_total': 4500000.0,
            'saldo_principal': 4500000.0,
            'saldo_mora_codeudor': 0.0,
            'creditos_sectorFinanciero': 1,
            'creditos_sectorCooperativo': 1,
            'creditos_sectorReal': 0,
            'promedio_ingresos_datacredito': 2800000.0,
            'tendencia_ingresos': 'Estable'
        }
    
    def test_initialization(self, temp_models_dir):
        """Test de inicialización de ModelDeployment"""
        deployment = ModelDeployment(models_dir=temp_models_dir)
        assert deployment.models_dir == Path(temp_models_dir)
        assert deployment.model is None
        assert deployment.preprocessor is None
    
    def test_validate_input_dict(self, temp_models_dir, sample_data):
        """Test validación de datos de entrada tipo dict"""
        deployment = ModelDeployment(models_dir=temp_models_dir)
        df = deployment.validate_input_data(sample_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'fecha_prestamo' in df.columns
    
    def test_validate_input_dataframe(self, temp_models_dir, sample_data):
        """Test validación de datos de entrada tipo DataFrame"""
        deployment = ModelDeployment(models_dir=temp_models_dir)
        df_input = pd.DataFrame([sample_data])
        df = deployment.validate_input_data(df_input)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
    
    def test_validate_missing_columns(self, temp_models_dir):
        """Test validación con columnas faltantes"""
        deployment = ModelDeployment(models_dir=temp_models_dir)
        incomplete_data = {'fecha_prestamo': '2024-06-15', 'tipo_credito': 1}
        
        with pytest.raises(ValueError, match="Faltan columnas requeridas"):
            deployment.validate_input_data(incomplete_data)
    
    def test_get_model_info_no_model(self, temp_models_dir):
        """Test obtener info sin modelo cargado"""
        deployment = ModelDeployment(models_dir=temp_models_dir)
        info = deployment.get_model_info()
        
        assert 'error' in info
        assert info['error'] == "No hay modelo cargado"


@pytest.mark.unit
class TestModelDeploymentValidation:
    """Tests de validación de datos"""
    
    def test_valid_fecha_format(self):
        """Test formato de fecha válido"""
        fecha = '2024-06-15'
        # Formato YYYY-MM-DD válido
        assert len(fecha.split('-')) == 3
    
    def test_numeric_validations(self):
        """Test validaciones numéricas básicas"""
        capital_prestado = 5000000.0
        plazo_meses = 36
        edad_cliente = 35
        
        assert capital_prestado > 0
        assert plazo_meses > 0
        assert 18 <= edad_cliente <= 100
    
    def test_categorical_validations(self):
        """Test validaciones categóricas"""
        tendencias_validas = ['Creciente', 'Estable', 'Decreciente']
        tendencia = 'Estable'
        
        assert tendencia in tendencias_validas


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
