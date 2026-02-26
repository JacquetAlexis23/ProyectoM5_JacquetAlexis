"""
Model Deployment Module - Versión 1.0.0
=======================================
Módulo para gestión, guardado, carga y predicción de modelos ML.
Preparado para despliegue en API con FastAPI.

Funcionalidades:
- Guardado de modelo y pipeline de preprocesamiento
- Carga de modelo entrenado
- Predicciones individuales y por lotes
- Validación de datos de entrada
- Registro de métricas y versiones

Autor: Alexis Jacquet - Experto Data Science
Proyecto: M5 - Henry - Avance 3
Fecha: Febrero 2026
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union
import json
import warnings
warnings.filterwarnings('ignore')


class ModelDeployment:
    """
    Clase principal para despliegue de modelos ML.
    Gestiona carga, predicción y versionado.
    """

    def __init__(self, models_dir: str = "../../models"):
        """
        Inicializa el deployment manager

        Args:
            models_dir: Directorio donde se guardan los modelos
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.preprocessor = None
        self.date_engineer = None
        self.financial_engineer = None
        self.feature_names = None
        self.model_info = {}

    # pylint: disable-next=too-many-positional-arguments
    def save_model(self,
                   model,
                   preprocessor,
                   date_engineer,
                   financial_engineer,
                   feature_names,
                   model_name: str,
                   metrics: Dict = None,
                   version: str = None):
        """
        Guarda el modelo y todos sus componentes necesarios

        Args:
            model: Modelo entrenado
            preprocessor: Pipeline de preprocesamiento
            date_engineer: Transformer de features de fecha
            financial_engineer: Transformer de features financieras
            feature_names: Nombres de las características
            model_name: Nombre del modelo
            metrics: Diccionario con métricas del modelo
            version: Versión del modelo
        """
        print("\n" + "="*80)
        print("💾 GUARDANDO MODELO PARA PRODUCCIÓN")
        print("="*80)

        # Generar versión automática si no se proporciona
        if version is None:
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Crear directorio de versión
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Guardar componentes
        model_path = version_dir / 'model.joblib'
        preprocessor_path = version_dir / 'preprocessor.joblib'
        date_engineer_path = version_dir / 'date_engineer.joblib'
        financial_engineer_path = version_dir / 'financial_engineer.joblib'
        features_path = version_dir / 'feature_names.joblib'

        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        joblib.dump(date_engineer, date_engineer_path)
        joblib.dump(financial_engineer, financial_engineer_path)
        joblib.dump(feature_names, features_path)

        print(f"✓ Modelo guardado: {model_path}")
        print(f"✓ Preprocessor guardado: {preprocessor_path}")
        print(f"✓ Date Engineer guardado: {date_engineer_path}")
        print(f"✓ Financial Engineer guardado: {financial_engineer_path}")
        print(f"✓ Feature names guardados: {features_path}")

        # Guardar metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'n_features': len(feature_names),
            'metrics': metrics if metrics else {}
        }

        metadata_path = version_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        print(f"✓ Metadata guardada: {metadata_path}")

        # Crear symlink a 'latest' (versión más reciente)
        latest_path = self.models_dir / 'latest'
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()

        # En Windows, podemos copiar en lugar de symlink si hay problemas de permisos
        try:
            latest_path.symlink_to(version_dir.absolute(), target_is_directory=True)
            print("✓ Enlace 'latest' actualizado")
        except OSError:
            # Fallback: guardar la ruta en un archivo de texto
            with open(self.models_dir / 'latest.txt', 'w', encoding='utf-8') as f:
                f.write(str(version_dir.absolute()))
            print("✓ Referencia 'latest' guardada (archivo)")

        print("\n" + "="*80)
        print(f"✅ MODELO GUARDADO EXITOSAMENTE - Versión: {version}")
        print("="*80)

        return version_dir

    def load_model(self, version: str = 'latest'):
        """
        Carga el modelo y todos sus componentes

        Args:
            version: Versión a cargar ('latest' para la más reciente)
        """
        # Validar el parámetro version para prevenir path traversal
        import re  # noqa: PLC0415
        if version != 'latest' and not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$', version):
            raise ValueError(
                f"Nombre de versión inválido: '{version}'. "
                "Solo se permiten caracteres alfanuméricos, guiones, puntos y underscores."
            )

        print("\n" + "="*80)
        print(f"📦 CARGANDO MODELO - Versión: {version}")
        print("="*80)

        # Determinar directorio de versión
        if version == 'latest':
            version_dir = self.models_dir / 'latest'

            # Si el symlink no funciona, leer del archivo de texto
            if not version_dir.exists() and (self.models_dir / 'latest.txt').exists():
                with open(self.models_dir / 'latest.txt', 'r', encoding='utf-8') as f:
                    version_dir = Path(f.read().strip())
        else:
            version_dir = self.models_dir / version

        if not version_dir.exists():
            raise FileNotFoundError(f"No se encontró el modelo versión '{version}' en {self.models_dir}")

        # Cargar componentes (joblib usa pickle internamente; los archivos provienen
        # exclusivamente del directorio interno de modelos, no de entrada del usuario)
        self.model = joblib.load(version_dir / 'model.joblib')  # nosonar
        self.preprocessor = joblib.load(version_dir / 'preprocessor.joblib')  # nosonar
        self.date_engineer = joblib.load(version_dir / 'date_engineer.joblib')  # nosonar
        self.financial_engineer = joblib.load(version_dir / 'financial_engineer.joblib')  # nosonar
        self.feature_names = joblib.load(version_dir / 'feature_names.joblib')  # nosonar

        # Cargar metadata
        with open(version_dir / 'metadata.json', 'r', encoding='utf-8') as f:
            self.model_info = json.load(f)

        print(f"✓ Modelo cargado: {self.model_info['model_name']}")
        print(f"✓ Tipo: {self.model_info['model_type']}")
        print(f"✓ Features: {self.model_info['n_features']}")
        print(f"✓ Fecha entrenamiento: {self.model_info['timestamp']}")

        if self.model_info.get('metrics'):
            print("\nℹﺏ  Métricas del modelo:")
            for metric, value in self.model_info['metrics'].items():
                if isinstance(value, (int, float)) and metric not in ['TN', 'FP', 'FN', 'TP']:
                    print(f"   • {metric}: {value:.4f}")

        print("\n" + "="*80)
        print("✅ MODELO CARGADO EXITOSAMENTE")
        print("="*80)

        return self

    def validate_input_data(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Valida y prepara los datos de entrada

        Args:
            data: Datos de entrada (dict o DataFrame)

        Returns:
            DataFrame validado y preparado
        """
        # Convertir dict a DataFrame si es necesario
        if isinstance(data, dict):
            # Si es un solo registro (verificar de forma segura con next/iter)
            first_value = next(iter(data.values()), None)
            if not isinstance(first_value, list):
                data = {k: [v] for k, v in data.items()}
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Columnas esperadas para predicción (sin Pago_atiempo)
        expected_columns = [
            'fecha_prestamo', 'tipo_credito', 'capital_prestado', 'plazo_meses',
            'edad_cliente', 'tipo_laboral', 'salario_cliente', 'total_otros_prestamos',
            'cuota_pactada', 'puntaje', 'puntaje_datacredito', 'cant_creditosvigentes',
            'huella_consulta', 'saldo_mora', 'saldo_total', 'saldo_principal',
            'saldo_mora_codeudor', 'creditos_sectorFinanciero', 'creditos_sectorCooperativo',
            'creditos_sectorReal', 'promedio_ingresos_datacredito', 'tendencia_ingresos'
        ]

        # Verificar columnas faltantes
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas: {missing_cols}")

        return df[expected_columns]

    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Aplica las transformaciones necesarias a los datos

        Args:
            df: DataFrame con datos crudos

        Returns:
            Array con features procesadas
        """
        # Aplicar transformaciones en el orden correcto
        X_dated = self.date_engineer.transform(df)
        X_engineered = self.financial_engineer.transform(X_dated)
        X_processed = self.preprocessor.transform(X_engineered)

        # Manejar NaN e infinitos
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)

        return X_processed

    def predict(self, data: Union[Dict, pd.DataFrame], return_proba: bool = False) -> Dict:
        """
        Realiza predicciones sobre los datos de entrada

        Args:
            data: Datos de entrada
            return_proba: Si retornar probabilidades

        Returns:
            Diccionario con predicciones y metadata
        """
        if self.model is None:
            raise ValueError("Modelo no cargado. Use load_model() primero.")

        # Validar y preparar datos
        df_validated = self.validate_input_data(data)

        # Preprocesar
        X_processed = self.preprocess_data(df_validated)

        # Predecir
        predictions = self.model.predict(X_processed)

        # Preparar respuesta
        result = {
            'predictions': predictions.tolist(),
            'n_samples': len(predictions),
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_info.get('version', 'unknown')
        }

        # Añadir probabilidades si se solicita
        if return_proba and hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_processed)
            result['probabilities'] = probas.tolist()
            result['probability_class_1'] = probas[:, 1].tolist()

        return result

    def predict_batch(self, data_list: List[Dict], return_proba: bool = False) -> Dict:
        """
        Realiza predicciones por lotes

        Args:
            data_list: Lista de registros
            return_proba: Si retornar probabilidades

        Returns:
            Diccionario con resultados del batch
        """
        # Convertir lista de dicts a DataFrame
        df = pd.DataFrame(data_list)

        return self.predict(df, return_proba=return_proba)

    def get_model_info(self) -> Dict:
        """Retorna información del modelo cargado"""
        if not self.model_info:
            return {"error": "No hay modelo cargado"}
        return self.model_info


# pylint: disable-next=too-many-positional-arguments
def deploy_best_model(models_dict: Dict,
                      results_df: pd.DataFrame,
                      preprocessor,
                      date_engineer,
                      financial_engineer,
                      feature_names,
                      models_dir: str = "../../models") -> ModelDeployment:
    """
    Función auxiliar para desplegar el mejor modelo desde el entrenamiento

    Args:
        models_dict: Diccionario con modelos entrenados
        results_df: DataFrame con resultados de evaluación
        preprocessor: Pipeline de preprocesamiento
        date_engineer: Transformer de fechas
        financial_engineer: Transformer financiero
        feature_names: Nombres de features
        models_dir: Directorio destino

    Returns:
        ModelDeployment: Instancia con modelo desplegado
    """
    # Obtener mejor modelo
    best_model_row = results_df.iloc[0]
    best_model_name = best_model_row['Model']
    best_model = models_dict[best_model_name]

    # Extraer métricas
    metrics = best_model_row.to_dict()

    # Crear deployment manager
    deployment = ModelDeployment(models_dir=models_dir)

    # Guardar modelo
    deployment.save_model(
        model=best_model,
        preprocessor=preprocessor,
        date_engineer=date_engineer,
        financial_engineer=financial_engineer,
        feature_names=feature_names,
        model_name=best_model_name,
        metrics=metrics
    )

    return deployment


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ℹ️  MODULE DEPLOYMENT - Listo para uso en producción")
    print("="*80)
    print("\nEste módulo debe usarse junto con el pipeline de entrenamiento:")
    print("\nEjemplo de uso:")
    print("""
    # 1. Guardar modelo después del entrenamiento
    from model_deploy import deploy_best_model

    deployment = deploy_best_model(
        models_dict=models,
        results_df=results,
        preprocessor=preprocessor,
        date_engineer=date_engineer,
        financial_engineer=financial_engineer,
        feature_names=feature_names
    )

    # 2. Cargar modelo para predicciones
    from model_deploy import ModelDeployment

    model = ModelDeployment()
    model.load_model('latest')

    # 3. Hacer predicciones
    data = {
        'fecha_prestamo': '2024-06-15',
        'tipo_credito': 1,
        'capital_prestado': 5000000,
        # ... más campos
    }

    resultado = model.predict(data, return_proba=True)
    print(resultado)
    """)
    print("="*80)
