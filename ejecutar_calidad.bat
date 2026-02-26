@echo off
REM ============================================================================
REM Script para ejecutar análisis de calidad de código
REM ============================================================================
REM Ejecuta Pylint y Flake8 para validar código
REM Autor: Alexis Jacquet - HENRY M5 Avance 3 Extra Credit
REM ============================================================================

echo ============================================================================
echo  ANALISIS DE CALIDAD DE CODIGO
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
echo [1/4] Activando entorno virtual...
call .venv\Scripts\activate.bat

REM Instalar herramientas si no estan
echo.
echo [2/4] Verificando herramientas de analisis...
pip install pylint flake8 --quiet

REM Ejecutar Pylint
echo.
echo [3/4] Ejecutando Pylint...
echo ============================================================================
pylint api_main.py train_and_deploy.py test_api.py mlops_pipeline/src/*.py --output-format=text --reports=y > pylint-report.txt
type pylint-report.txt
echo.
echo Reporte completo guardado en: pylint-report.txt
echo.

REM Ejecutar Flake8
echo.
echo [4/4] Ejecutando Flake8...
echo ============================================================================
flake8 api_main.py train_and_deploy.py test_api.py mlops_pipeline/src/*.py --output-file=flake8-report.txt --statistics
type flake8-report.txt
echo.
echo Reporte completo guardado en: flake8-report.txt
echo.

echo ============================================================================
echo  ANALISIS COMPLETADO
echo ============================================================================
echo.
echo Reportes generados:
echo   - pylint-report.txt
echo   - flake8-report.txt
echo.
echo Revisa los reportes para ver sugerencias de mejora
echo.
pause
