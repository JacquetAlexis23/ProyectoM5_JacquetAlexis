@echo off
REM ============================================================================
REM Script para probar la API
REM ============================================================================
REM Ejecuta tests automatizados de la API
REM Autor: Alexis Jacquet - HENRY M5 Avance 3
REM ============================================================================

echo ============================================================================
echo  TESTING DE LA API
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
echo [INFO] Activando entorno virtual...
call .venv\Scripts\activate.bat

echo.
echo [INFO] Ejecutando tests...
echo.

REM Ejecutar tests
python test_api.py

echo.
pause
