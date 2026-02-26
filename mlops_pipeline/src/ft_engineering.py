"""
Feature Engineering Pipeline - Versión 1.1.0
============================================
Pipeline robusto para ingeniería de características que incluye:
- Imputación de valores nulos
- Transformación de variables categóricas
- Feature engineering de fechas
- Escalado de variables numéricas
- Pipelines modulares y reutilizables
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Columnas que pueden causar fuga de información (data leakage).
# Usar drop_leakage=True en prepare_features() para eliminarlas antes del
# entrenamiento. La lista es configurable pasando leakage_cols a la función.
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS = ['puntaje']


class DateFeatureEngineer:
    """Clase para crear features a partir de fechas"""

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        return self

    def transform(self, X):
        X_copy = X.copy()
        if 'fecha_prestamo' in X_copy.columns:
            X_copy['fecha_prestamo'] = pd.to_datetime(X_copy['fecha_prestamo'])
            X_copy['mes_prestamo'] = X_copy['fecha_prestamo'].dt.month
            X_copy['dia_semana'] = X_copy['fecha_prestamo'].dt.dayofweek
            X_copy['trimestre'] = X_copy['fecha_prestamo'].dt.quarter
            X_copy['dias_desde_epoca'] = (X_copy['fecha_prestamo'] - pd.Timestamp('2024-01-01')).dt.days
            X_copy = X_copy.drop('fecha_prestamo', axis=1)
        return X_copy

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class FinancialFeatureEngineer:
    """Clase para crear features financieras derivadas"""

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Ratios financieros (usando np.where para evitar divisiones por cero)
        X_copy['ratio_cuota_salario'] = np.where(
            X_copy['salario_cliente'] > 0,
            X_copy['cuota_pactada'] / X_copy['salario_cliente'],
            0
        )
        X_copy['ratio_deuda_ingreso'] = np.where(
            X_copy['salario_cliente'] > 0,
            X_copy['saldo_total'] / X_copy['salario_cliente'],
            0
        )
        X_copy['ratio_capital_salario'] = np.where(
            X_copy['salario_cliente'] > 0,
            X_copy['capital_prestado'] / X_copy['salario_cliente'],
            0
        )

        # Indicadores de riesgo
        X_copy['tiene_mora'] = (X_copy['saldo_mora'] > 0).astype(int)
        X_copy['tiene_codeudor_mora'] = (X_copy['saldo_mora_codeudor'] > 0).astype(int)

        X_copy['nivel_endeudamiento'] = np.where(
            X_copy['salario_cliente'] > 0,
            X_copy['cant_creditosvigentes'] * X_copy['saldo_total'] / X_copy['salario_cliente'],
            0
        )

        # Distribución de créditos
        X_copy['total_creditos_sector'] = (X_copy['creditos_sectorFinanciero'] +
                                            X_copy['creditos_sectorCooperativo'] +
                                            X_copy['creditos_sectorReal'])

        X_copy['ratio_financiero'] = np.where(
            X_copy['total_creditos_sector'] > 0,
            X_copy['creditos_sectorFinanciero'] / X_copy['total_creditos_sector'],
            0
        )

        # Indicadores de capacidad de pago
        X_copy['ingreso_disponible'] = (X_copy['salario_cliente']
                                         - X_copy['cuota_pactada']
                                         - X_copy['total_otros_prestamos'])

        X_copy['ratio_puntajes'] = np.where(
            X_copy['puntaje_datacredito'] > 0,
            X_copy['puntaje'] / X_copy['puntaje_datacredito'],
            1
        )

        # Reemplazar infinitos y NaN que puedan haber quedado
        X_copy = X_copy.replace([np.inf, -np.inf], np.nan)

        return X_copy

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def create_preprocessing_pipeline(drop_cols=None):
    """
    Crea el pipeline completo de preprocesamiento siguiendo el diagrama de ColumnTransformer

    Args:
        drop_cols: Lista de columnas a excluir del preprocesador (p.ej. leakage).
                   Si es None no se excluye ninguna.

    Returns:
        ColumnTransformer configurado
    """

    # Variables numéricas
    numeric_features = [
        'tipo_credito', 'capital_prestado', 'plazo_meses', 'edad_cliente',
        'salario_cliente', 'total_otros_prestamos', 'cuota_pactada', 'puntaje',
        'puntaje_datacredito', 'cant_creditosvigentes', 'huella_consulta',
        'saldo_mora', 'saldo_total', 'saldo_principal', 'saldo_mora_codeudor',
        'creditos_sectorFinanciero', 'creditos_sectorCooperativo',
        'creditos_sectorReal', 'promedio_ingresos_datacredito'
    ]

    # Excluir columnas marcadas como leakage (si las hay)
    if drop_cols:
        numeric_features = [f for f in numeric_features if f not in drop_cols]

    # Variables categóricas nominales (OneHotEncoder)
    categorical_nominal = ['tipo_laboral']

    # Variables categóricas ordinales (OrdinalEncoder)
    categorical_ordinal = ['tendencia_ingresos']
    ordinal_categories = [['Decreciente', 'Estable', 'Creciente']]

    # Features derivadas de fecha
    date_features = ['mes_prestamo', 'dia_semana', 'trimestre', 'dias_desde_epoca']

    # Features financieras derivadas
    financial_features = [
        'ratio_cuota_salario', 'ratio_deuda_ingreso', 'ratio_capital_salario',
        'tiene_mora', 'tiene_codeudor_mora', 'nivel_endeudamiento',
        'total_creditos_sector', 'ratio_financiero', 'ingreso_disponible',
        'ratio_puntajes'
    ]

    # Pipeline para variables numéricas
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para variables categóricas nominales
    categorical_nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Pipeline para variables categóricas ordinales
    categorical_ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Estable')),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Pipeline para features de fecha (ya numéricas, solo escalar)
    date_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Pipeline para features financieras (ya numéricas, solo escalar)
    financial_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # ColumnTransformer que combina todos los pipelines
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat_nominal', categorical_nominal_pipeline, categorical_nominal),
            ('cat_ordinal', categorical_ordinal_pipeline, categorical_ordinal),
            ('date', date_pipeline, date_features),
            ('financial', financial_pipeline, financial_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return column_transformer


def prepare_features(df, target_column='Pago_atiempo',
                     drop_leakage=False, leakage_cols=None):
    """
    Prepara las características aplicando feature engineering

    Args:
        df: DataFrame con los datos originales
        target_column: Nombre de la columna objetivo
        drop_leakage: Si True, elimina columnas con fuga de información antes
                      de ajustar el preprocesador. No afecta a features
                      derivadas (p.ej. ratio_puntajes sigue siendo calculada).
        leakage_cols: Lista personalizada de columnas a eliminar cuando
                      drop_leakage=True. Si None se usa LEAKAGE_COLUMNS.

    Returns:
        tuple: (X_processed, y, feature_names, preprocessor,
                date_engineer, financial_engineer)
    """
    # Separar features y target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Aplicar feature engineering de fechas
    date_engineer = DateFeatureEngineer()
    X_with_dates = date_engineer.fit_transform(X)

    # Aplicar feature engineering financiero.
    # Nota: ratio_puntajes se calcula aquí usando 'puntaje'; si luego se
    # elimina 'puntaje' (raw), la feature derivada se conserva igualmente.
    financial_engineer = FinancialFeatureEngineer()
    X_with_features = financial_engineer.fit_transform(X_with_dates)

    # Eliminar columnas de leakage DESPUÉS del feature engineering
    # para preservar las features derivadas (p.ej. ratio_puntajes).
    dropped = []
    if drop_leakage:
        requested = leakage_cols if leakage_cols is not None else LEAKAGE_COLUMNS
        dropped = [c for c in requested if c in X_with_features.columns]
        if dropped:
            X_with_features = X_with_features.drop(dropped, axis=1)
            print(f"⚠️  Leakage columns eliminadas: {dropped}")

    # Crear y aplicar pipeline de preprocesamiento
    preprocessor = create_preprocessing_pipeline(drop_cols=dropped if dropped else None)
    X_processed = preprocessor.fit_transform(X_with_features)

    # Verificar y eliminar NaN/Inf remanentes
    # Reemplazar cualquier NaN con 0 y infinitos con valores grandes finitos
    X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)

    # Obtener nombres de features
    feature_names = preprocessor.get_feature_names_out()

    return X_processed, y, feature_names, preprocessor, date_engineer, financial_engineer


def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Divide los datos en conjuntos de entrenamiento y prueba

    Args:
        X: Features procesadas
        y: Variable objetivo
        test_size: Proporción del conjunto de prueba
        random_state: Semilla para reproducibilidad
        stratify: Variable para estratificación

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if stratify is not None:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    return X_train, X_test, y_train, y_test


def get_feature_engineering_pipeline():
    """
    Retorna el pipeline completo de feature engineering para usar en producción

    Returns:
        Pipeline completo de transformación
    """
    date_engineer = DateFeatureEngineer()
    financial_engineer = FinancialFeatureEngineer()
    preprocessor = create_preprocessing_pipeline()

    # Pipeline completo
    full_pipeline = Pipeline([
        ('date_features', date_engineer),
        ('financial_features', financial_engineer),
        ('preprocessing', preprocessor)
    ])

    return full_pipeline


def load_and_prepare_data(file_path, test_size=0.2, random_state=42,
                          drop_leakage=False):
    """
    Función principal que carga y prepara los datos para entrenamiento

    Args:
        file_path: Ruta al archivo CSV
        test_size: Proporción del conjunto de prueba
        random_state: Semilla para reproducibilidad
        drop_leakage: Si True elimina columnas en LEAKAGE_COLUMNS antes de
                      entrenar el preprocesador (ver prepare_features).

    Returns:
        dict: Diccionario con todos los componentes necesarios
    """
    print("="*80)
    print("PIPELINE DE FEATURE ENGINEERING - VERSIÓN 1.1.0")
    print("="*80)

    if drop_leakage:
        print(f"\n⚠️  Modo drop_leakage=True → se eliminarán: {LEAKAGE_COLUMNS}")

    # Cargar datos
    print("\n📂 Cargando datos...")
    df = pd.read_csv(file_path)
    print(f"✓ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")

    # Preparar features
    print("\n🔧 Aplicando feature engineering...")
    X_processed, y, feature_names, preprocessor, date_engineer, financial_engineer = prepare_features(
        df, drop_leakage=drop_leakage
    )
    print(f"✓ Features generadas: {X_processed.shape[1]} características")

    # Dividir datos
    print("\n✂️ Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = split_data(X_processed, y, test_size, random_state, stratify=y)
    print(f"✓ Entrenamiento: {X_train.shape[0]} registros")
    print(f"✓ Prueba: {X_test.shape[0]} registros")

    # Información de balance de clases
    print("\n📊 Balance de clases:")
    print(f"   Entrenamiento - Clase 0: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
    print(f"   Entrenamiento - Clase 1: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
    print(f"   Prueba - Clase 0: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.1f}%)")
    print(f"   Prueba - Clase 1: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.1f}%)")

    print("\n✅ Feature engineering completado exitosamente!")
    print("="*80)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'preprocessor': preprocessor,
        'date_engineer': date_engineer,
        'financial_engineer': financial_engineer,
        'full_pipeline': get_feature_engineering_pipeline()
    }


if __name__ == "__main__":
    # Ejemplo de uso
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, "data", "Base_de_datos.csv")

    result = load_and_prepare_data(data_path)

    print(f"\n🎯 Features finales: {len(result['feature_names'])}")
    print("\nPrimeras 10 características:")
    for i, name in enumerate(result['feature_names'][:10], 1):
        print(f"  {i}. {name}")
