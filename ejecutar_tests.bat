@echo off
REM ============================================================================
REM Script para ejecutar tests con cobertura
REM ============================================================================
REM Ejecuta pytest con cobertura de código
REM Autor: Alexis Jacquet - HENRY M5 Avance 3 Extra Credit
REM ============================================================================

echo ============================================================================
echo  EJECUTANDO TESTS CON COBERTURA DE CODIGO
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
echo [1/3] Activando entorno virtual...
call .venv\Scripts\activate.bat

REM Instalar dependencias de testing si no estan
echo.
echo [2/3] Verificando dependencias de testing...
pip install pytest pytest-cov --quiet

REM Ejecutar tests
echo.
echo [3/3] Ejecutando tests...
echo.
pytest tests/ -v --cov=. --cov-report=html --cov-report=xml --cov-report=term-missing --cov-branch

REM Verificar resultado
if %errorlevel% neq 0 (
    echo.
    echo [ADVERTENCIA] Algunos tests fallaron
) else (
    echo.
    echo [EXITO] Todos los tests pasaron
)

echo.
echo ============================================================================
echo  REPORTE DE COBERTURA GENERADO
echo ============================================================================
echo.
echo Reportes generados:
echo   - HTML: htmlcov\index.html
echo   - XML:  coverage.xml
echo.
echo Abre htmlcov\index.html en tu navegador para ver el reporte detallado
echo.
pause
