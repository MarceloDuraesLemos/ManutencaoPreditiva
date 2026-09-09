@echo off
title Plataforma de Manutencao Preditiva
cd /d "%~dp0"

echo ============================================
echo   PLATAFORMA DE MANUTENCAO PREDITIVA
echo ============================================
echo.
echo Iniciando sistema...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERRO: Ambiente virtual .venv nao encontrado.
    echo.
    echo Execute a instalacao do projeto antes de iniciar.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run "app\app.py"

pause