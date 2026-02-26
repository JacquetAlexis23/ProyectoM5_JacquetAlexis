"""
FastAPI ML Model Deployment - API Principal
===========================================
API REST para disponibilizar modelo de predicción de pagos.
Soporta predicciones individuales y por lotes.

Endpoints:
- GET /: Información de la API
- GET /health: Health check
- GET /model/info: Información del modelo
- POST /predict: Predicción individual
- POST /predict/batch: Predicción por lotes

Autor: Alexis Jacquet - Experto Data Science
Proyecto: M5 - Henry - Avance 3
Fecha: Febrero 2026
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import sys
from pathlib import Path

# Añadir path del módulo de deployment
sys.path.insert(0, str(Path(__file__).parent / "mlops_pipeline" / "src"))  # noqa: E402

from model_deploy import ModelDeployment  # pylint: disable=wrong-import-position


# ============================================================================
# MODELOS PYDANTIC PARA VALIDACIÓN DE DATOS
# ============================================================================

class ClienteData(BaseModel):
    """Modelo de datos para un cliente/préstamo individual"""

    fecha_prestamo: str = Field(..., description="Fecha del préstamo (YYYY-MM-DD)")
    tipo_credito: int = Field(..., ge=0, description="Tipo de crédito")
    capital_prestado: float = Field(..., gt=0, description="Capital prestado")
    plazo_meses: int = Field(..., gt=0, description="Plazo en meses")
    edad_cliente: int = Field(..., ge=18, le=100, description="Edad del cliente")
    tipo_laboral: str = Field(..., description="Tipo de empleo")
    salario_cliente: float = Field(..., ge=0, description="Salario del cliente")
    total_otros_prestamos: float = Field(..., ge=0, description="Total otros préstamos")
    cuota_pactada: float = Field(..., gt=0, description="Cuota pactada")
    puntaje: float = Field(..., ge=0, description="Puntaje interno")
    puntaje_datacredito: float = Field(..., ge=0, description="Puntaje DataCrédito")
    cant_creditosvigentes: int = Field(..., ge=0, description="Cantidad de créditos vigentes")
    huella_consulta: int = Field(..., ge=0, description="Huella de consulta")
    saldo_mora: float = Field(..., ge=0, description="Saldo en mora")
    saldo_total: float = Field(..., ge=0, description="Saldo total")
    saldo_principal: float = Field(..., ge=0, description="Saldo principal")
    saldo_mora_codeudor: float = Field(..., ge=0, description="Saldo mora codeudor")
    creditos_sectorFinanciero: int = Field(..., ge=0, description="Créditos sector financiero")
    creditos_sectorCooperativo: int = Field(..., ge=0, description="Créditos sector cooperativo")
    creditos_sectorReal: int = Field(..., ge=0, description="Créditos sector real")
    promedio_ingresos_datacredito: float = Field(..., ge=0, description="Promedio ingresos DataCrédito")
    tendencia_ingresos: str = Field(..., description="Tendencia de ingresos: Creciente, Estable, Decreciente")

    @validator('fecha_prestamo')  # pylint: disable=no-self-argument
    def validate_fecha(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError as exc:
            raise ValueError('La fecha debe estar en formato YYYY-MM-DD') from exc

    @validator('tendencia_ingresos')  # pylint: disable=no-self-argument
    def validate_tendencia(cls, v):
        if v not in ['Creciente', 'Estable', 'Decreciente']:
            raise ValueError('tendencia_ingresos debe ser: Creciente, Estable o Decreciente')
        return v

    class Config:
        schema_extra = {
            "example": {
                "fecha_prestamo": "2024-06-15",
                "tipo_credito": 1,
                "capital_prestado": 5000000.0,
                "plazo_meses": 36,
                "edad_cliente": 35,
                "tipo_laboral": "Empleado",
                "salario_cliente": 3000000.0,
                "total_otros_prestamos": 500000.0,
                "cuota_pactada": 180000.0,
                "puntaje": 750.0,
                "puntaje_datacredito": 720.0,
                "cant_creditosvigentes": 2,
                "huella_consulta": 5,
                "saldo_mora": 0.0,
                "saldo_total": 4500000.0,
                "saldo_principal": 4500000.0,
                "saldo_mora_codeudor": 0.0,
                "creditos_sectorFinanciero": 1,
                "creditos_sectorCooperativo": 1,
                "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": 2800000.0,
                "tendencia_ingresos": "Estable"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Modelo para predicciones por lotes"""
    data: List[ClienteData] = Field(..., description="Lista de registros para predecir")
    return_probabilities: bool = Field(default=False, description="Retornar probabilidades")

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "fecha_prestamo": "2024-06-15",
                        "tipo_credito": 1,
                        "capital_prestado": 5000000.0,
                        "plazo_meses": 36,
                        "edad_cliente": 35,
                        "tipo_laboral": "Empleado",
                        "salario_cliente": 3000000.0,
                        "total_otros_prestamos": 500000.0,
                        "cuota_pactada": 180000.0,
                        "puntaje": 750.0,
                        "puntaje_datacredito": 720.0,
                        "cant_creditosvigentes": 2,
                        "huella_consulta": 5,
                        "saldo_mora": 0.0,
                        "saldo_total": 4500000.0,
                        "saldo_principal": 4500000.0,
                        "saldo_mora_codeudor": 0.0,
                        "creditos_sectorFinanciero": 1,
                        "creditos_sectorCooperativo": 1,
                        "creditos_sectorReal": 0,
                        "promedio_ingresos_datacredito": 2800000.0,
                        "tendencia_ingresos": "Estable"
                    }
                ],
                "return_probabilities": True
            }
        }


class PredictionResponse(BaseModel):
    """Modelo de respuesta de predicción"""
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    probability_class_1: Optional[List[float]] = None
    n_samples: int
    timestamp: str
    model_version: str
    interpretation: Optional[List[str]] = None


# ============================================================================
# INICIALIZACIÓN DE LA API Y MODELO
# ============================================================================

app = FastAPI(
    title="API de Predicción de Pagos a Tiempo",
    description="""
    API para predicción de pagos a tiempo usando Machine Learning.

    ## Características

    * **Predicción individual**: Predice si un cliente pagará a tiempo
    * **Predicción por lotes**: Procesa múltiples registros simultáneamente
    * **Probabilidades**: Opcionalmente retorna probabilidades de predicción
    * **Versionado**: Soporta múltiples versiones de modelos
    * **Health check**: Monitoreo del estado de la API

    ## Etiquetas de predicción

    * `0`: Cliente NO pagará a tiempo (riesgo de mora)
    * `1`: Cliente SÍ pagará a tiempo (bajo riesgo)

    ## Autor

    Alexis Jacquet - Data Science Expert
    HENRY M5 - Avance 3
    """,
    version="1.0.0",
    contact={
        "name": "Alexis Jacquet",
        "email": "alexis.jacquet@example.com"
    }
)

# Variable global para el modelo
model_deployment = None


@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar la API"""
    global model_deployment

    print("\n" + "="*80)
    print("🚀 INICIANDO API DE PREDICCIÓN")
    print("="*80)

    try:
        # Determinar ruta de modelos
        models_dir = Path(__file__).parent / "models"

        # Crear deployment manager y cargar modelo
        model_deployment = ModelDeployment(models_dir=str(models_dir))
        model_deployment.load_model('latest')

        print("\n✅ API lista para recibir peticiones")
        print("="*80 + "\n")

    except FileNotFoundError as e:
        print("\n⚠️  ADVERTENCIA: No se encontró modelo entrenado")
        print("   Por favor, entrena un modelo primero usando run_pipeline.py")
        print(f"   Error: {e}")
        print("="*80 + "\n")
    except Exception as e:  # pylint: disable=broad-exception-caught  # NOSONAR - startup handler, all errors must be caught
        print(f"\n❌ ERROR al cargar modelo: {e}")
        print("="*80 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la API"""
    print("\n" + "="*80)
    print("🛑 CERRANDO API DE PREDICCIÓN")
    print("="*80 + "\n")


# ============================================================================
# ENDPOINTS DE LA API
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """
    Endpoint raíz - Información de la API
    """
    return {
        "api": "API de Predicción de Pagos a Tiempo",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "author": "Alexis Jacquet",
        "project": "HENRY M5 - Avance 3"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """
    Health check - Verificar estado de la API y modelo
    """
    model_loaded = model_deployment is not None and model_deployment.model is not None

    return {
        "status": "healthy" if model_loaded else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "api_version": "1.0.0"
    }


@app.get("/model/info", tags=["Modelo"])
async def get_model_info():
    """
    Obtener información del modelo cargado
    """
    if model_deployment is None or model_deployment.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Por favor entrena un modelo primero."
        )

    return model_deployment.get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict_single(cliente: ClienteData, return_probabilities: bool = False):
    """
    Realizar predicción para un solo cliente

    - **cliente**: Datos del cliente/préstamo
    - **return_probabilities**: Si retornar probabilidades (default: False)

    Retorna:
    - `predictions`: [0] o [1]
    - `probability_class_1`: Probabilidad de pagar a tiempo (si se solicita)
    - `interpretation`: Interpretación de la predicción
    """
    if model_deployment is None or model_deployment.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Por favor entrena un modelo primero."
        )

    try:
        # Convertir a dict
        data_dict = cliente.dict()

        # Realizar predicción
        result = model_deployment.predict(data_dict, return_proba=return_probabilities)

        # Añadir interpretación
        prediction = result['predictions'][0]
        interpretation = [
            "✅ Cliente pagará a tiempo (bajo riesgo)" if prediction == 1
            else "⚠️ Cliente NO pagará a tiempo (alto riesgo de mora)"
        ]
        result['interpretation'] = interpretation

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error en los datos de entrada: {str(e)}"
        ) from e
    except Exception as e:  # pylint: disable=broad-except  # NOSONAR - API endpoint, re-raised as HTTPException
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la predicción: {str(e)}"
        ) from e


@app.post("/predict/batch", response_model=PredictionResponse, tags=["Predicción"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Realizar predicciones para múltiples clientes (batch)

    - **data**: Lista de registros de clientes
    - **return_probabilities**: Si retornar probabilidades

    Retorna:
    - `predictions`: Lista de predicciones [0, 1, 1, ...]
    - `probability_class_1`: Probabilidades por registro
    - `interpretation`: Interpretación de cada predicción
    """
    if model_deployment is None or model_deployment.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Por favor entrena un modelo primero."
        )

    if len(request.data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La lista de datos está vacía"
        )

    try:
        # Convertir a lista de dicts
        data_list = [cliente.dict() for cliente in request.data]

        # Realizar predicción por lotes
        result = model_deployment.predict_batch(data_list, return_proba=request.return_probabilities)

        # Añadir interpretación para cada predicción
        interpretations = [
            "✅ Pagará a tiempo" if pred == 1 else "⚠️ NO pagará a tiempo"
            for pred in result['predictions']
        ]
        result['interpretation'] = interpretations

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error en los datos de entrada: {str(e)}"
        ) from e
    except Exception as e:  # pylint: disable=broad-except  # NOSONAR - API endpoint, re-raised as HTTPException
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la predicción: {str(e)}"
        ) from e


# ============================================================================
# PUNTO DE ENTRADA PARA EJECUCIÓN DIRECTA
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*80)
    print("🚀 INICIANDO SERVIDOR API")
    print("="*80)
    print("\n📍 La API estará disponible en: http://127.0.0.1:8000")
    print("📚 Documentación interactiva en: http://127.0.0.1:8000/docs")
    print("📖 Documentación alternativa en: http://127.0.0.1:8000/redoc")
    print("\n" + "="*80 + "\n")

    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
