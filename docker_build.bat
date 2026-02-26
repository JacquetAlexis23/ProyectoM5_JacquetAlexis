@echo off
REM ============================================================================
REM Script para construir imagen Docker
REM ============================================================================
REM Construye la imagen Docker de la API
REM Autor: Alexis Jacquet - HENRY M5 Avance 3
REM ============================================================================

echo ============================================================================
echo  CONSTRUCCION DE IMAGEN DOCKER
echo ============================================================================
echo.

REM Verificar si Docker está instalado
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker no esta instalado o no esta en el PATH
    echo Por favor instala Docker Desktop para Windows
    echo https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Verificar si existe el modelo
if not exist models\latest (
    if not exist models\latest.txt (
        echo [ADVERTENCIA] No se encontro modelo entrenado
        echo Se recomienda entrenar el modelo primero: entrenar_modelo.bat
        echo.
        choice /C SN /M "Deseas continuar de todas formas?"
        if errorlevel 2 exit /b 0
        echo.
    )
)

echo [1/3] Construyendo imagen Docker...
echo.
docker build -t ml-payment-prediction-api:latest .

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Fallo la construccion de la imagen
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Verificando imagen creada...
docker images ml-payment-prediction-api:latest

echo.
echo ============================================================================
echo  IMAGEN DOCKER CREADA EXITOSAMENTE
echo ============================================================================
echo.
echo Comandos utiles:
echo.
echo  Iniciar contenedor:
echo    docker run -d -p 8000:8000 --name ml-api ml-payment-prediction-api:latest
echo.
echo  Ver logs:
echo    docker logs -f ml-api
echo.
echo  Detener contenedor:
echo    docker stop ml-api
echo.
echo  Eliminar contenedor:
echo    docker rm ml-api
echo.
echo  O usar Docker Compose:
echo    docker-compose up -d
echo.
echo ============================================================================
echo.
pause
