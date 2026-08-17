@echo off
setlocal

for %%I in ("%~dp0.") do set "SCARLET_DIR=%%~fI"
set "VENV_DIR=%SCARLET_DIR%\scarlet_venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

set "TEMPLATE_NOTEBOOK=%SCARLET_DIR%\notebooks\tutorial.ipynb"
set "SESSIONS_DIR=%SCARLET_DIR%\tutorial_sessions"

title Tutoriel SCARLET

if not exist "%PYTHON_EXE%" (
    echo.
    echo ERREUR : environnement Python introuvable.
    echo Lancez d'abord install_windows.cmd.
    echo.
    echo %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

if not exist "%TEMPLATE_NOTEBOOK%" (
    echo.
    echo ERREUR : notebook modele introuvable.
    echo.
    echo %TEMPLATE_NOTEBOOK%
    echo.
    pause
    exit /b 1
)

if not exist "%SESSIONS_DIR%" (
    mkdir "%SESSIONS_DIR%"
)

:MENU
cls
echo.
echo ==================================================
echo               Tutoriel SCARLET
echo ==================================================
echo.
echo   1 - Creer un nouveau notebook
echo.
echo   2 - Ouvrir un notebook existant
echo.
echo   3 - Quitter
echo.
echo ==================================================
echo.

set /p "CHOIX=Votre choix : "

if "%CHOIX%"=="1" goto NOUVEAU
if "%CHOIX%"=="2" goto EXISTANT
if "%CHOIX%"=="3" goto FIN

echo.
echo Choix invalide.
pause
goto MENU

:NOUVEAU
cls
echo.
echo Entrez votre nom ou vos initiales.
echo.

set "USER_NAME="
set /p "USER_NAME=Nom ou initiales : "

if not defined USER_NAME (
    set "USER_NAME=utilisateur"
)

set "USER_NAME=%USER_NAME: =_%"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%I"

set "USER_DIR=%SESSIONS_DIR%\%USER_NAME%"

if not exist "%USER_DIR%" (
    mkdir "%USER_DIR%"
)

set "USER_NOTEBOOK=%USER_DIR%\tutorial_%USER_NAME%_%TIMESTAMP%.ipynb"

copy "%TEMPLATE_NOTEBOOK%" "%USER_NOTEBOOK%" >nul

if errorlevel 1 (
    echo.
    echo ERREUR : impossible de copier le notebook.
    echo.
    pause
    goto MENU
)

echo.
echo Ouverture du nouveau notebook :
echo.
echo %USER_NOTEBOOK%
echo.

cd /d "%USER_DIR%"

"%PYTHON_EXE%" -m jupyter lab "%USER_NOTEBOOK%"

goto MENU

:EXISTANT
cls
echo.
echo Ouverture de JupyterLab dans le dossier :
echo.
echo %SESSIONS_DIR%
echo.
echo Selectionnez ensuite votre notebook dans
echo l'explorateur de fichiers de JupyterLab.
echo.

cd /d "%SESSIONS_DIR%"

"%PYTHON_EXE%" -m jupyter lab .

goto MENU

:FIN
endlocal
exit /b 0
