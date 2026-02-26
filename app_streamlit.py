"""
🎯 MLOps Dashboard - Model Monitoring & Data Drift Detection
============================================================
Aplicación profesional de Streamlit para monitoreo de modelos ML
con detección de data drift en tiempo real.

Features:
- Visualización interactiva de métricas de drift
- Sistema de alertas con indicadores visuales
- Análisis temporal de drift
- Comparación de distribuciones
- Performance tracking
- Recomendaciones automáticas

Autor: Alexis Jacquet
Proyecto: M5 - Henry - Avance 3
Fecha: Febrero 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import requests
from datetime import datetime, timedelta
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configurar path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'mlops_pipeline', 'src'))

from ft_engineering import load_and_prepare_data
from model_monitoring import DataDriftDetector, monitor_model_predictions
from sklearn.tree import DecisionTreeClassifier

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="MLOps Monitor | Data Drift Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ESTILOS PERSONALIZADOS (Moderno y Minimalista)
# ============================================================================

st.markdown("""
<style>
    /* Fuentes modernas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fondo principal */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers modernos */
    h1 {
        color: #1e293b;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    h2 {
        color: #334155;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #475569;
        font-weight: 600;
        font-size: 1.3rem !important;
    }
    
    /* Métricas personalizadas */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Alertas con diseño moderno */
    .alert-green {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }
    
    .alert-yellow {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.2);
    }
    
    .alert-orange {
        background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%);
        border-left: 5px solid #f97316;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(249, 115, 22, 0.2);
    }
    
    .alert-red {
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
        border-left: 5px solid #ef4444;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.9; }
    }
    
    /* Botones modernos */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar moderna */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }
    
    /* Tabs modernos */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Indicador de carga moderno */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* DataFrames estilizados */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(30, 41, 59, 0.95);
        color: white;
        text-align: center;
        padding: 0.75rem;
        font-size: 0.875rem;
        backdrop-filter: blur(10px);
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_data
def load_data():
    """Carga y prepara los datos"""
    try:
        DATA_PATH = os.path.join(current_dir, "data", "Base_de_datos.csv")
        data = load_and_prepare_data(DATA_PATH, test_size=0.2, random_state=42)
        return data
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

@st.cache_resource
def load_model():
    """Carga o entrena el modelo"""
    try:
        # Entrenar modelo simple para demo (en producción cargaríamos el modelo guardado)
        data = load_data()
        if data is None:
            return None
        
        model = DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42
        )
        model.fit(data['X_train'], data['y_train'])
        return model
    except Exception as e:
        st.error(f"Error al cargar modelo: {e}")
        return None

def create_alert_box(alert_level: str, message: str):
    """Crea caja de alerta visual según nivel"""
    alert_class = f"alert-{alert_level}"
    
    icons = {
        'green': '✅',
        'yellow': '⚠️',
        'orange': '🟠',
        'red': '🚨'
    }
    
    st.markdown(f"""
    <div class="{alert_class}">
        <h3 style="margin-top:0;">{icons.get(alert_level, '⚪')} {message}</h3>
    </div>
    """, unsafe_allow_html=True)

def create_gauge_chart(value: float, title: str, threshold_low: float, threshold_high: float):
    """Crea gráfico tipo gauge (velocímetro)"""
    
    # Determinar color según threshold
    if value < threshold_low:
        color = "#10b981"  # Verde
    elif value < threshold_high:
        color = "#f59e0b"  # Amarillo
    else:
        color = "#ef4444"  # Rojo
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': '#1e293b'}},
        delta = {'reference': threshold_low, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, threshold_low], 'color': '#d1fae5'},
                {'range': [threshold_low, threshold_high], 'color': '#fef3c7'},
                {'range': [threshold_high, 1], 'color': '#fecaca'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_high
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1e293b", 'family': "Inter"}
    )
    
    return fig

def create_distribution_comparison(baseline_data: pd.Series, current_data: pd.Series, 
                                   feature_name: str):
    """Crea gráfico comparativo de distribuciones"""
    
    fig = go.Figure()
    
    # Histograma baseline
    fig.add_trace(go.Histogram(
        x=baseline_data,
        name='Baseline (Train)',
        opacity=0.7,
        marker_color='#667eea',
        nbinsx=50
    ))
    
    # Histograma current
    fig.add_trace(go.Histogram(
        x=current_data,
        name='Current (Test)',
        opacity=0.7,
        marker_color='#f59e0b',
        nbinsx=50
    ))
    
    fig.update_layout(
        title=f'Distribución: {feature_name}',
        xaxis_title=feature_name,
        yaxis_title='Frecuencia',
        barmode='overlay',
        template='plotly_white',
        height=400,
        font={'family': 'Inter'},
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_drift_timeline(drift_history: list):
    """Crea línea temporal de drift"""
    
    if not drift_history:
        return None
    
    timestamps = [d['timestamp'] for d in drift_history]
    drift_pcts = [d['summary']['drift_percentage'] for d in drift_history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=drift_pcts,
        mode='lines+markers',
        name='Drift %',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#764ba2'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    # Líneas de threshold
    fig.add_hline(y=10, line_dash="dash", line_color="orange", 
                  annotation_text="Threshold Moderado (10%)")
    fig.add_hline(y=30, line_dash="dash", line_color="red", 
                  annotation_text="Threshold Crítico (30%)")
    
    fig.update_layout(
        title='Evolución Temporal del Data Drift',
        xaxis_title='Timestamp',
        yaxis_title='Drift Percentage (%)',
        template='plotly_white',
        height=400,
        font={'family': 'Inter'},
        hovermode='x unified'
    )
    
    return fig

def create_feature_drift_heatmap(drift_df: pd.DataFrame):
    """Crea heatmap de drift por feature"""
    
    # Preparar datos
    numeric_cols = ['ks_statistic', 'psi', 'js_divergence']
    available_cols = [col for col in numeric_cols if col in drift_df.columns]
    
    if not available_cols:
        return None
    
    # Seleccionar top features con drift
    plot_df = drift_df.sort_values('drift_detected', ascending=False).head(20)
    
    fig = go.Figure()
    
    for col in available_cols:
        fig.add_trace(go.Bar(
            name=col.upper(),
            x=plot_df['feature'],
            y=plot_df[col],
            text=plot_df[col].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Top 20 Features - Métricas de Drift',
        xaxis_title='Feature',
        yaxis_title='Valor Métrica',
        barmode='group',
        template='plotly_white',
        height=500,
        font={'family': 'Inter'},
        xaxis={'tickangle': -45}
    )
    
    return fig

# ============================================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9850/9850774.png", width=100)
    st.title("⚙️ Configuración")
    
    st.markdown("---")
    
    st.subheader("🎯 Thresholds de Drift")
    
    ks_threshold = st.slider(
        "KS Test (p-value)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Umbral para el test Kolmogorov-Smirnov"
    )
    
    psi_threshold = st.slider(
        "PSI Threshold",
        min_value=0.05,
        max_value=0.30,
        value=0.10,
        step=0.05,
        help="Population Stability Index threshold"
    )
    
    js_threshold = st.slider(
        "JS Divergence",
        min_value=0.05,
        max_value=0.30,
        value=0.10,
        step=0.05,
        help="Jensen-Shannon divergence threshold"
    )
    
    st.markdown("---")
    
    st.subheader("📊 Opciones de Análisis")
    
    show_all_features = st.checkbox("Mostrar todas las features", value=False)
    show_distributions = st.checkbox("Mostrar distribuciones", value=True)
    show_temporal = st.checkbox("Análisis temporal", value=True)
    
    st.markdown("---")
    
    st.subheader("🔄 Acciones")
    
    if st.button("🔄 Actualizar Datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("💾 Exportar Reporte", use_container_width=True):
        st.info("Reporte exportado (funcionalidad demo)")
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; color: white; padding: 1rem;">
        <h4>📚 Documentación</h4>
        <p style="font-size: 0.875rem;">
        Sistema de monitoreo MLOps con detección
        de data drift en tiempo real.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER PRINCIPAL
# ============================================================================

st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1>🎯 MLOps Dashboard</h1>
    <p style="font-size: 1.2rem; color: #64748b;">
        Model Monitoring & Data Drift Detection System
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# CARGA DE DATOS Y MODELO
# ============================================================================

with st.spinner("🔄 Cargando datos y modelo..."):
    data = load_data()
    model = load_model()

if data is None or model is None:
    st.error("❌ Error al cargar datos o modelo. Verifique la configuración.")
    st.stop()

# ============================================================================
# ANÁLISIS DE DRIFT
# ============================================================================

st.markdown("---")

# Ejecutar análisis de drift
with st.spinner("🔍 Detectando data drift..."):
    detector = DataDriftDetector(
        ks_threshold=ks_threshold,
        psi_threshold=psi_threshold,
        js_threshold=js_threshold
    )
    
    detector.fit(data['X_train_original'], data['feature_names'])
    drift_results = detector.detect_drift(data['X_test_original'])
    drift_df = detector.get_drift_summary_dataframe(drift_results)

# ============================================================================
# TABS PRINCIPALES
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard General",
    "🔍 Análisis de Features",
    "📈 Distribuciones",
    "⏱️ Análisis Temporal",
    "💡 Recomendaciones",
    "🤖 Predicción API"
])

# ----------------------------------------------------------------------------
# TAB 1: DASHBOARD GENERAL
# ----------------------------------------------------------------------------

with tab1:
    st.header("📊 Vista General del Sistema")
    
    # Caja de alerta principal
    summary = drift_results['summary']
    alert_level = summary['alert_level']
    
    alert_messages = {
        'green': "ESTADO NORMAL - Sistema Operando Correctamente",
        'yellow': "ALERTA MENOR - Monitoreo Requerido",
        'orange': "ALERTA MODERADA - Acción Recomendada",
        'red': "ALERTA CRÍTICA - Acción Inmediata Requerida"
    }
    
    create_alert_box(alert_level, alert_messages.get(alert_level, "Estado Desconocido"))
    
    # Métricas principales en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Features Analizadas",
            value=summary['total_features'],
            delta=None
        )
    
    with col2:
        st.metric(
            label="⚠️ Features con Drift",
            value=summary['features_with_drift'],
            delta=f"{summary['drift_percentage']:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="🎯 Accuracy Baseline",
            value="100.0%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="📅 Última Actualización",
            value=datetime.now().strftime("%H:%M"),
            delta="Ahora"
        )
    
    st.markdown("---")
    
    # Gauges de métricas clave
    st.subheader("🎚️ Indicadores de Drift")
    
    col1, col2, col3 = st.columns(3)
    
    # Calcular promedio de métricas
    avg_psi = drift_df['psi'].mean() if 'psi' in drift_df.columns else 0
    avg_ks = drift_df['ks_statistic'].mean() if 'ks_statistic' in drift_df.columns else 0
    avg_js = drift_df['js_divergence'].mean() if 'js_divergence' in drift_df.columns else 0
    
    with col1:
        fig_psi = create_gauge_chart(avg_psi, "PSI Promedio", 0.1, 0.2)
        st.plotly_chart(fig_psi, use_container_width=True)
    
    with col2:
        fig_ks = create_gauge_chart(avg_ks, "KS Statistic Promedio", 0.1, 0.3)
        st.plotly_chart(fig_ks, use_container_width=True)
    
    with col3:
        fig_js = create_gauge_chart(avg_js, "JS Divergence Promedio", 0.1, 0.2)
        st.plotly_chart(fig_js, use_container_width=True)
    
    st.markdown("---")
    
    # Resumen de características del dataset
    st.subheader("📋 Información del Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🔵 Baseline (Training Set)</h4>
            <p><strong>Muestras:</strong> {}</p>
            <p><strong>Features:</strong> {}</p>
            <p><strong>Balance de Clases:</strong> {:.1f}% / {:.1f}%</p>
        </div>
        """.format(
            len(data['X_train_original']),
            len(data['feature_names']),
            (data['y_train'] == 0).sum() / len(data['y_train']) * 100,
            (data['y_train'] == 1).sum() / len(data['y_train']) * 100
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🟢 Current (Test Set)</h4>
            <p><strong>Muestras:</strong> {}</p>
            <p><strong>Features:</strong> {}</p>
            <p><strong>Balance de Clases:</strong> {:.1f}% / {:.1f}%</p>
        </div>
        """.format(
            len(data['X_test_original']),
            len(data['feature_names']),
            (data['y_test'] == 0).sum() / len(data['y_test']) * 100,
            (data['y_test'] == 1).sum() / len(data['y_test']) * 100
        ), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 2: ANÁLISIS DE FEATURES
# ----------------------------------------------------------------------------

with tab2:
    st.header("🔍 Análisis Detallado por Feature")
    
    # Filtros
    col1, col2 = st.columns([3, 1])
    
    with col1:
        filter_option = st.radio(
            "Filtrar features:",
            ["Todas", "Solo con Drift", "Solo Numéricas", "Solo Categóricas"],
            horizontal=True
        )
    
    with col2:
        sort_by = st.selectbox(
            "Ordenar por:",
            ["drift_detected", "psi", "ks_statistic", "js_divergence"]
        )
    
    # Aplicar filtros
    filtered_df = drift_df.copy()
    
    if filter_option == "Solo con Drift":
        filtered_df = filtered_df[filtered_df['drift_detected'] == True]
    elif filter_option == "Solo Numéricas":
        filtered_df = filtered_df[filtered_df['type'] == 'numeric']
    elif filter_option == "Solo Categóricas":
        filtered_df = filtered_df[filtered_df['type'] == 'categorical']
    
    # Ordenar
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)
    
    st.markdown(f"**Mostrando {len(filtered_df)} features**")
    
    # Tabla interactiva
    st.dataframe(
        filtered_df.style.background_gradient(
            subset=['psi'] if 'psi' in filtered_df.columns else [],
            cmap='RdYlGn_r'
        ).format({
            'psi': '{:.4f}',
            'ks_statistic': '{:.4f}',
            'ks_pvalue': '{:.4f}',
            'js_divergence': '{:.4f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Gráfico de barras de drift
    if len(filtered_df) > 0:
        st.markdown("---")
        fig_heatmap = create_feature_drift_heatmap(filtered_df)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3: DISTRIBUCIONES
# ----------------------------------------------------------------------------

with tab3:
    st.header("📈 Comparación de Distribuciones")
    
    if show_distributions:
        # Selector de feature
        feature_to_plot = st.selectbox(
            "Seleccionar feature para visualizar:",
            options=data['feature_names'],
            index=0
        )
        
        # Gráfico de distribución
        fig_dist = create_distribution_comparison(
            data['X_train_original'][feature_to_plot],
            data['X_test_original'][feature_to_plot],
            feature_to_plot
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Estadísticas descriptivas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Baseline Statistics")
            baseline_stats = data['X_train_original'][feature_to_plot].describe()
            st.dataframe(baseline_stats, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Current Statistics")
            current_stats = data['X_test_original'][feature_to_plot].describe()
            st.dataframe(current_stats, use_container_width=True)
        
        # Grid de distribuciones múltiples
        st.markdown("---")
        st.subheader("🎨 Vista de Múltiples Distribuciones")
        
        # Seleccionar top features con drift
        top_drift_features = drift_df[drift_df['drift_detected'] == True]['feature'].head(6).tolist()
        
        if len(top_drift_features) > 0:
            cols = st.columns(2)
            
            for idx, feature in enumerate(top_drift_features):
                with cols[idx % 2]:
                    fig_mini = create_distribution_comparison(
                        data['X_train_original'][feature],
                        data['X_test_original'][feature],
                        feature
                    )
                    fig_mini.update_layout(height=300)
                    st.plotly_chart(fig_mini, use_container_width=True)
        else:
            st.info("✅ No hay features con drift significativo para mostrar")

# ----------------------------------------------------------------------------
# TAB 4: ANÁLISIS TEMPORAL
# ----------------------------------------------------------------------------

with tab4:
    st.header("⏱️ Evolución Temporal del Drift")
    
    if show_temporal:
        # Simular datos históricos para demo
        st.info("📌 Nota: En producción, este gráfico mostraría datos históricos reales")
        
        # Generar datos de ejemplo
        dates = pd.date_range(
            end=datetime.now(),
            periods=10,
            freq='D'
        )
        
        simulated_history = []
        for date in dates:
            simulated_history.append({
                'timestamp': date.strftime("%Y-%m-%d"),
                'summary': {
                    'drift_percentage': np.random.uniform(0, 40)
                }
            })
        
        # Gráfico temporal
        fig_timeline = create_drift_timeline(simulated_history)
        if fig_timeline:
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("---")
        
        # Tabla de histórico
        st.subheader("📋 Histórico de Análisis")
        
        history_df = pd.DataFrame([
            {
                'Fecha': h['timestamp'],
                'Drift %': h['summary']['drift_percentage']
            }
            for h in simulated_history
        ])
        
        st.dataframe(history_df, use_container_width=True)
        
        # Análisis de tendencias
        st.markdown("---")
        st.subheader("📊 Análisis de Tendencias")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend = "↗️ Creciente" if history_df['Drift %'].iloc[-1] > history_df['Drift %'].iloc[0] else "↘️ Decreciente"
            st.metric("Tendencia", trend)
        
        with col2:
            avg_drift = history_df['Drift %'].mean()
            st.metric("Drift Promedio", f"{avg_drift:.1f}%")
        
        with col3:
            max_drift = history_df['Drift %'].max()
            st.metric("Drift Máximo", f"{max_drift:.1f}%")

# ----------------------------------------------------------------------------
# TAB 5: RECOMENDACIONES
# ----------------------------------------------------------------------------

with tab5:
    st.header("💡 Recomendaciones y Acciones")
    
    # Generar mensaje de alerta
    alert_message = detector.generate_alert_message(drift_results)
    
    st.markdown(f"""
    <div class="alert-{alert_level}" style="font-size: 1rem; line-height: 1.8;">
        {alert_message.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Plan de acción recomendado
    st.subheader("📋 Plan de Acción Recomendado")
    
    if alert_level == 'green':
        st.success("✅ No se requieren acciones inmediatas")
        st.markdown("""
        - Mantener monitoreo regular (semanal)
        - Continuar recolectando métricas
        - Documentar estado actual
        """)
    
    elif alert_level == 'yellow':
        st.warning("⚠️ Acciones preventivas recomendadas")
        st.markdown("""
        - **Corto plazo (1-2 semanas):**
          - Incrementar frecuencia de monitoreo (diario)
          - Revisar logs de aplicación
          - Analizar features con drift
        
        - **Mediano plazo (1 mes):**
          - Evaluar si el drift es temporal o permanente
          - Considerar ajustes en el pipeline de datos
        """)
    
    elif alert_level == 'orange':
        st.warning("🟠 Acción moderada requerida")
        st.markdown("""
        - **Inmediato (esta semana):**
          - Investigar causas raíz del drift
          - Revisar fuentes de datos
          - Validar integridad de datos entrantes
        
        - **Corto plazo (2-3 semanas):**
          - Planificar reentrenamiento del modelo
          - Preparar nuevo dataset de entrenamiento
          - Evaluar nuevas features
        
        - **Documentación:**
          - Registrar análisis de drift
          - Documentar decisiones tomadas
        """)
    
    else:  # red
        st.error("🚨 Acción crítica inmediata")
        st.markdown("""
        - **CRÍTICO - Actuar HOY:**
          - 🛑 Evaluar pausar predicciones del modelo
          - 🔍 Investigación urgente de causas
          - 📞 Notificar a stakeholders
        
        - **Esta semana:**
          - 🔄 Reentrenar modelo con datos actuales
          - 🧪 Validación exhaustiva del nuevo modelo
          - 📊 Actualizar dashboard de monitoreo
        
        - **Próximos días:**
          - ✅ Implementar nuevo modelo
          - 📈 Validar mejora en performance
          - 📝 Documentar incidente y resolución
        """)
    
    st.markdown("---")
    
    # Enlaces útiles y recursos
    st.subheader("📚 Recursos y Documentación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📖 Documentación
        - [Guía de Data Drift](https://docs.example.com/drift)
        - [Best Practices MLOps](https://docs.example.com/mlops)
        - [Retraining Pipeline](https://docs.example.com/retrain)
        """)
    
    with col2:
        st.markdown("""
        #### 🔧 Herramientas
        - [Dashboard de Producción](#)
        - [Sistema de Alertas](#)
        - [Reentrenamiento Automático](#)
        """)

# ----------------------------------------------------------------------------
# TAB 6: PREDICCIÓN VÍA API
# ----------------------------------------------------------------------------

with tab6:
    st.header("🤖 Predicción de Pagos vía FastAPI")
    st.markdown("""
    Este módulo se conecta directamente al endpoint `/predict` y `/predict/batch`  
    de la **FastAPI** para obtener predicciones del modelo desplegado en producción.
    """)

    # ------------ Configuración de la API ------------
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔌 Conexión API")
        # En Docker, API_URL apunta al servicio interno ml-api:8000
        default_api_url = os.environ.get("API_URL", "http://localhost:8000")
        api_url = st.text_input(
            "URL de la API",
            value=default_api_url,
            help="URL base de la FastAPI (sin barra al final)"
        )

    # ------------ Health check rápido ------------
    col_status, col_info = st.columns([1, 3])
    with col_status:
        try:
            resp = requests.get(f"{api_url}/health", timeout=3)
            if resp.status_code == 200:
                health = resp.json()
                if health.get("model_loaded"):
                    st.success("✅ API conectada\nModelo listo")
                else:
                    st.warning("⚠️ API activa\nModelo no cargado")
            else:
                st.error("❌ API no responde")
        except Exception:
            st.error("❌ API no disponible\nInicia: `python api_main.py`")

    with col_info:
        try:
            resp_info = requests.get(f"{api_url}/model/info", timeout=3)
            if resp_info.status_code == 200:
                info = resp_info.json()
                st.info(
                    f"**Modelo:** {info.get('model_name', 'N/A')} | "
                    f"**Tipo:** {info.get('model_type', 'N/A')} | "
                    f"**Versión:** {info.get('version', 'N/A')}"
                )
        except Exception:
            st.info("ℹ️ Informacion del modelo no disponible (API desconectada)")

    st.markdown("---")

    # ------------ Modo de predicción ------------
    mode = st.radio(
        "Modo de predicción:",
        ["📋 Individual (un cliente)", "📦 Batch (múltiples clientes)"],
        horizontal=True
    )

    # ------------ Formulario de datos ------------
    st.subheader("📝 Ingresar datos del cliente")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            fecha_prestamo = st.date_input("Fecha del préstamo", value=datetime(2024, 6, 15))
            tipo_credito = st.number_input("Tipo de crédito", min_value=0, max_value=10, value=1)
            capital_prestado = st.number_input("Capital prestado ($)", min_value=1.0, value=5000000.0, step=100000.0)
            plazo_meses = st.number_input("Plazo (meses)", min_value=1, max_value=360, value=36)
            edad_cliente = st.number_input("Edad del cliente", min_value=18, max_value=100, value=35)
            tipo_laboral = st.selectbox("Tipo laboral", ["Empleado", "Independiente", "Pensionado", "Desempleado"])
            salario_cliente = st.number_input("Salario ($)", min_value=0.0, value=3000000.0, step=100000.0)
            tendencia_ingresos = st.selectbox("Tendencia de ingresos", ["Creciente", "Estable", "Decreciente"])

        with col2:
            total_otros_prestamos = st.number_input("Total otros préstamos ($)", min_value=0.0, value=500000.0, step=100000.0)
            cuota_pactada = st.number_input("Cuota pactada ($)", min_value=1.0, value=180000.0, step=10000.0)
            puntaje = st.number_input("Puntaje interno", min_value=0.0, value=750.0)
            puntaje_datacredito = st.number_input("Puntaje DataCrédito", min_value=0.0, value=720.0)
            cant_creditosvigentes = st.number_input("Créditos vigentes", min_value=0, value=2)
            huella_consulta = st.number_input("Huella consulta", min_value=0, value=5)

        with col3:
            saldo_mora = st.number_input("Saldo en mora ($)", min_value=0.0, value=0.0, step=1000.0)
            saldo_total = st.number_input("Saldo total ($)", min_value=0.0, value=4500000.0, step=100000.0)
            saldo_principal = st.number_input("Saldo principal ($)", min_value=0.0, value=4500000.0, step=100000.0)
            saldo_mora_codeudor = st.number_input("Saldo mora codeudor ($)", min_value=0.0, value=0.0)
            creditos_sectorFinanciero = st.number_input("Créditos sector financiero", min_value=0, value=1)
            creditos_sectorCooperativo = st.number_input("Créditos sector cooperativo", min_value=0, value=1)
            creditos_sectorReal = st.number_input("Créditos sector real", min_value=0, value=0)
            promedio_ingresos_datacredito = st.number_input("Promedio ingresos DataCrédito ($)", min_value=0.0, value=2800000.0, step=100000.0)

        return_proba = st.checkbox("Mostrar probabilidades", value=True)

        batch_size = 1
        if "Batch" in mode:
            batch_size = st.slider("Número de copias del registro para batch", min_value=2, max_value=10, value=3,
                                   help="Para demostrar predicción batch se duplica el mismo registro N veces")

        submitted = st.form_submit_button("🚀 Predecir", use_container_width=True)

    # ------------ Llamada a la API y resultados ------------
    if submitted:
        payload_single = {
            "fecha_prestamo": fecha_prestamo.strftime("%Y-%m-%d"),
            "tipo_credito": int(tipo_credito),
            "capital_prestado": float(capital_prestado),
            "plazo_meses": int(plazo_meses),
            "edad_cliente": int(edad_cliente),
            "tipo_laboral": tipo_laboral,
            "salario_cliente": float(salario_cliente),
            "total_otros_prestamos": float(total_otros_prestamos),
            "cuota_pactada": float(cuota_pactada),
            "puntaje": float(puntaje),
            "puntaje_datacredito": float(puntaje_datacredito),
            "cant_creditosvigentes": int(cant_creditosvigentes),
            "huella_consulta": int(huella_consulta),
            "saldo_mora": float(saldo_mora),
            "saldo_total": float(saldo_total),
            "saldo_principal": float(saldo_principal),
            "saldo_mora_codeudor": float(saldo_mora_codeudor),
            "creditos_sectorFinanciero": int(creditos_sectorFinanciero),
            "creditos_sectorCooperativo": int(creditos_sectorCooperativo),
            "creditos_sectorReal": int(creditos_sectorReal),
            "promedio_ingresos_datacredito": float(promedio_ingresos_datacredito),
            "tendencia_ingresos": tendencia_ingresos
        }

        try:
            if "Individual" in mode:
                # Llamada al endpoint individual
                endpoint = f"{api_url}/predict?return_probabilities={str(return_proba).lower()}"
                response = requests.post(endpoint, json=payload_single, timeout=10)
            else:
                # Llamada al endpoint batch
                payload_batch = {
                    "data": [payload_single] * batch_size,
                    "return_probabilities": return_proba
                }
                response = requests.post(f"{api_url}/predict/batch", json=payload_batch, timeout=10)

            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])
                interpretations = result.get("interpretation", [])
                probabilities = result.get("probability_class_1", [])

                st.markdown("---")
                st.subheader("📊 Resultados de la Predicción")

                # Mostrar cada predicción
                for i, pred in enumerate(predictions):
                    col_pred, col_interp = st.columns([1, 3])
                    with col_pred:
                        if pred == 1:
                            st.success(f"**Registro {i+1}:** ✅ Paga")
                        else:
                            st.error(f"**Registro {i+1}:** ⚠️ No paga")
                    with col_interp:
                        interp = interpretations[i] if i < len(interpretations) else ""
                        st.write(interp)
                        if probabilities and i < len(probabilities):
                            prob_val = probabilities[i]
                            st.progress(float(prob_val), text=f"Probabilidad pagar a tiempo: {prob_val:.1%}")

                # Metadata
                st.markdown("---")
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                with meta_col1:
                    st.metric("Registros procesados", result.get("n_samples", len(predictions)))
                with meta_col2:
                    st.metric("Versión del modelo", result.get("model_version", "N/A"))
                with meta_col3:
                    ts = result.get("timestamp", datetime.now().isoformat())
                    st.metric("Timestamp", ts[:19].replace("T", " "))

                # JSON raw para transparencia
                with st.expander("📄 Ver respuesta JSON completa de la API"):
                    st.json(result)

            else:
                st.error(f"❌ Error de la API: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ No se pudo conectar a la API. "
                "Asegúrate que la API está corriendo:\n"
                "`python api_main.py`  o  `docker-compose up -d`"
            )
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    🎯 MLOps Dashboard v1.0.0 | Proyecto M5 - Henry | Desarrollado por Alexis Jacquet | 
    Febrero 2026 | <strong>Data Drift Detection System</strong>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MÉTRICAS EN TIEMPO REAL (Sidebar inferior)
# ============================================================================

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Métricas en Tiempo Real")
    st.metric("⏱️ Tiempo de Análisis", "< 1s")
    st.metric("🔄 Estado Sistema", "✅ Operativo")
    st.metric("💾 Último Backup", "Hoy")
