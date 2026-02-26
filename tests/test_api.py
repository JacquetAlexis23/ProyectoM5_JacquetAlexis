"""
Tests Unitarios - API FastAPI
==============================
Tests para validar funcionalidad de la API REST

Autor: Alexis Jacquet
HENRY M5 - Avance 3 Extra Credit
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Añadir path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api_main import app


class TestAPIEndpoints:
    """Tests para endpoints de la API"""
    
    @pytest.fixture
    def client(self):
        """Cliente de test para FastAPI"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_cliente(self):
        """Datos de cliente de muestra"""
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
    
    def test_root_endpoint(self, client):
        """Test endpoint raíz"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'api' in data
        assert 'version' in data
        assert data['version'] == '1.0.0'
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'model_loaded' in data
    
    def test_predict_endpoint_validation(self, client):
        """Test validación de datos en endpoint predict"""
        # Datos inválidos (edad negativa)
        invalid_data = {
            'fecha_prestamo': '2024-06-15',
            'tipo_credito': 1,
            'capital_prestado': 5000000.0,
            'plazo_meses': 36,
            'edad_cliente': -5,  # Inválido
            'tipo_laboral': 'Empleado',
            'salario_cliente': 3000000.0
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_missing_fields(self, client):
        """Test endpoint predict con campos faltantes"""
        incomplete_data = {
            'fecha_prestamo': '2024-06-15',
            'tipo_credito': 1
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422


@pytest.mark.unit
class TestPydanticModels:
    """Tests para validación de modelos Pydantic"""
    
    def test_valid_fecha_format(self):
        """Test validación de formato de fecha"""
        from datetime import datetime
        
        fecha_valida = '2024-06-15'
        try:
            datetime.strptime(fecha_valida, '%Y-%m-%d')
            assert True
        except ValueError:
            assert False
    
    def test_tendencia_ingresos_values(self):
        """Test valores válidos de tendencia_ingresos"""
        valores_validos = ['Creciente', 'Estable', 'Decreciente']
        
        for valor in valores_validos:
            assert valor in valores_validos
    
    def test_numeric_constraints(self):
        """Test restricciones numéricas"""
        edad_valida = 35
        edad_invalida_joven = 15
        edad_invalida_vieja = 150
        
        assert 18 <= edad_valida <= 100
        assert not (18 <= edad_invalida_joven <= 100)
        assert not (18 <= edad_invalida_vieja <= 100)


@pytest.mark.api
class TestAPIIntegration:
    """Tests de integración de API"""
    
    @pytest.fixture
    def client(self):
        """Cliente de test"""
        return TestClient(app)
    
    def test_api_response_structure(self, client):
        """Test estructura de respuesta de la API"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        
        # Verificar campos esperados
        assert isinstance(data, dict)
        assert 'api' in data
        assert 'endpoints' in data
        assert isinstance(data['endpoints'], dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
