@echo off
setlocal

rem ==================================================
rem Configuration
rem ==================================================

set "SCARLET_DIR=C:\Users\gac-sansllb\Document\SCARLET"
set "VENV_DIR=%SCARLET_DIR%\scarlet_venv"

title SCARLET Viewer

rem ==================================================
rem Verifications
rem ==================================================

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo.
    echo ERREUR : environnement Python introuvable.
    echo.
    echo Chemin recherche :
    echo %VENV_DIR%
    echo.
    pause
    exit /b 1
)

rem ==================================================
rem Lancement
rem ==================================================

cd /d "%SCARLET_DIR%"

echo.
echo Lancement de SCARLET Viewer...
echo.

"%VENV_DIR%\Scripts\python.exe" -m scarlet viewer

if errorlevel 1 (
    echo.
    echo ERREUR : SCARLET Viewer n'a pas pu etre lance.
    echo.
    echo Testez eventuellement la commande suivante :
    echo "%VENV_DIR%\Scripts\scarlet.exe" viewer
    echo.
    pause
)

endlocal