@echo off
REM ============================================================================
REM Script para entrenar y desplegar el modelo
REM ============================================================================
REM Ejecuta el pipeline completo de entrenamiento y guardado del modelo
REM Autor: Alexis Jacquet - HENRY M5 Avance 3
REM ============================================================================

echo ============================================================================
echo  ENTRENAMIENTO Y DESPLIEGUE DE MODELO ML
echo ============================================================================
echo.

REM Verificar si existe el entorno virtual
if not exist .venv (
    echo [ERROR] No se encontro el entorno virtual .venv
    echo Por favor ejecuta set_up.bat primero
    pause
    exit /b 1
)

REM Activar entorno virtual
echo [1/2] Activando entorno virtual...
call .venv\Scripts\activate.bat

REM Ejecutar pipeline
echo.
echo [2/2] Ejecutando pipeline de entrenamiento y despliegue...
echo.
python train_and_deploy.py

REM Verificar resultado
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] El pipeline fallo con errores
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================================
echo  ENTRENAMIENTO COMPLETADO EXITOSAMENTE
echo ============================================================================
echo.
echo Ahora puedes:
echo   1. Iniciar la API: ejecutar_api.bat
echo   2. Probar la API: test_api.bat
echo   3. Construir imagen Docker: docker_build.bat
echo.
pause
