@echo off
setlocal

rem ==================================================
rem Configuration
rem ==================================================

set "SCARLET_DIR=C:\Users\gac-sansllb\Document\SCARLET"
set "VENV_DIR=%SCARLET_DIR%\scarlet_venv"

set "TEMPLATE_NOTEBOOK=%SCARLET_DIR%\scarlet\notebooks\tutorial.ipynb"
set "SESSIONS_DIR=%SCARLET_DIR%\tutorial_sessions"

title Tutoriel SCARLET

rem ==================================================
rem Verifications
rem ==================================================

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo.
    echo ERREUR : environnement Python introuvable.
    echo.
    echo %VENV_DIR%
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

rem ==================================================
rem Menu
rem ==================================================

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

rem ==================================================
rem Creer un nouveau notebook
rem ==================================================

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

rem Remplacer les espaces par des tirets bas
set "USER_NAME=%USER_NAME: =_%"

rem Creer une date et une heure
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

"%VENV_DIR%\Scripts\python.exe" -m jupyter lab "%USER_NOTEBOOK%"

goto MENU

rem ==================================================
rem Ouvrir un notebook existant
rem ==================================================

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

"%VENV_DIR%\Scripts\python.exe" -m jupyter lab .

goto MENU

rem ==================================================
rem Quitter
rem ==================================================

:FIN
endlocal
exit /b 0