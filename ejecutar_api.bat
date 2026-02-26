@echo off
REM ============================================================================
REM Script para ejecutar la API de predicción
REM ============================================================================
REM Inicia el servidor FastAPI en modo desarrollo
REM Autor: Alexis Jacquet - HENRY M5 Avance 3
REM ============================================================================

echo ============================================================================
echo  INICIANDO API DE PREDICCION ML
echo ============================================================================
echo.

REM Verificar si existe el entorno virtual
if not exist .venv (
    echo [ERROR] No se encontro el entorno virtual .venv
    echo Por favor ejecuta set_up.bat primero
    pause
    exit /b 1
)

REM Verificar si existe el modelo
if not exist models\latest (
    if not exist models\latest.txt (
        echo [ADVERTENCIA] No se encontro modelo entrenado
        echo Por favor ejecuta entrenar_modelo.bat primero
        echo.
        echo Continuando de todas formas...
        echo.
    )
)

REM Activar entorno virtual
echo [INFO] Activando entorno virtual...
call .venv\Scripts\activate.bat

echo.
echo ============================================================================
echo  SERVIDOR API INICIADO
echo ============================================================================
echo.
echo  URL Base:           http://localhost:8000
echo  Documentacion:      http://localhost:8000/docs
echo  Documentacion Alt:  http://localhost:8000/redoc
echo  Health Check:       http://localhost:8000/health
echo.
echo  Presiona Ctrl+C para detener el servidor
echo.
echo ============================================================================
echo.

REM Ejecutar API
python api_main.py

echo.
echo Servidor detenido.
pause
