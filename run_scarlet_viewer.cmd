@echo off
setlocal

set "SCARLET_DIR=C:\Users\gac-sansllb\Documents\SCARLET"
set "VENV_DIR=%SCARLET_DIR%\scarlet_venv"

title SCARLET Viewer

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo ERREUR : environnement virtuel introuvable.
    echo.
    echo Chemin recherche :
    echo %VENV_DIR%
    echo.
    pause
    exit /b 1
)

cd /d "%SCARLET_DIR%"

call "%VENV_DIR%\Scripts\activate.bat"

scarlet viewer

if errorlevel 1 (
    echo.
    echo ERREUR : SCARLET Viewer n'a pas pu etre lance.
    echo.
    pause
)

endlocal