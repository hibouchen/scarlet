@echo off
setlocal

for %%I in ("%~dp0.") do set "SCARLET_DIR=%%~fI"
set "VENV_DIR=%SCARLET_DIR%\scarlet_venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

title SCARLET Viewer

if not exist "%PYTHON_EXE%" (
    echo.
    echo ERREUR : environnement virtuel introuvable.
    echo Lancez d'abord install_windows.cmd.
    echo.
    echo Chemin recherche :
    echo %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

cd /d "%SCARLET_DIR%"

"%PYTHON_EXE%" -m scarlet.cli viewer %*

if errorlevel 1 (
    echo.
    echo ERREUR : SCARLET Viewer n'a pas pu etre lance.
    echo.
    pause
)

endlocal
