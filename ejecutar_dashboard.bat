@echo off
REM ============================================================================
REM Script de Ejecucion - Dashboard de Monitoreo MLOps
REM ============================================================================
REM Autor: Alexis Jacquet
REM Proyecto: M5 - Henry - Avance 3
REM ============================================================================

echo.
echo ================================================================================
echo                     DASHBOARD DE MONITOREO - MLOps PIPELINE
echo ================================================================================
echo.
echo Proyecto: Prediccion de Pagos con Data Drift Detection
echo Autor: Alexis Jacquet
echo ================================================================================
echo.

REM Verificar que Python este instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado o no esta en el PATH
    echo.
    echo Por favor instala Python desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python detectado
echo.

REM Verificar que Streamlit este instalado
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ADVERTENCIA] Streamlit no esta instalado
    echo.
    echo Instalando dependencias...
    pip install -r requirements.txt
    echo.
    if errorlevel 1 (
        echo [ERROR] No se pudieron instalar las dependencias
        pause
        exit /b 1
    )
)

echo [OK] Streamlit detectado
echo.

REM Verificar que el archivo de la app existe
if not exist "app_streamlit.py" (
    echo [ERROR] No se encuentra el archivo app_streamlit.py
    echo Asegurate de estar en el directorio correcto del proyecto
    pause
    exit /b 1
)

echo [OK] Archivo de aplicacion encontrado
echo.

echo ================================================================================
echo                         INICIANDO DASHBOARD...
echo ================================================================================
echo.
echo Dashboard URL: http://localhost:8501
echo.
echo Presiona Ctrl+C para detener el servidor
echo ================================================================================
echo.

REM Ejecutar Streamlit
streamlit run app_streamlit.py

pause
