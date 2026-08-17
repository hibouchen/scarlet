@echo off
setlocal

for %%I in ("%~dp0.") do set "SCARLET_DIR=%%~fI"
set "VENV_DIR=%SCARLET_DIR%\scarlet_venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_CANDIDATE_FILE=%TEMP%\scarlet_python_candidate.txt"

title Installation SCARLET pour Windows

echo.
echo ==================================================
echo         Installation SCARLET pour Windows
echo ==================================================
echo.
echo Dossier du projet :
echo %SCARLET_DIR%
echo.

del "%PYTHON_CANDIDATE_FILE%" >nul 2>&1

where py >nul 2>&1
if not errorlevel 1 (
    py -3.9 -c "import sys; print(sys.executable)" >"%PYTHON_CANDIDATE_FILE%" 2>nul
    if not errorlevel 1 goto PYTHON_FOUND

    py -3 -c "import sys; print(sys.executable)" >"%PYTHON_CANDIDATE_FILE%" 2>nul
    if not errorlevel 1 goto PYTHON_FOUND
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; print(sys.executable)" >"%PYTHON_CANDIDATE_FILE%" 2>nul
    if not errorlevel 1 goto PYTHON_FOUND
)

echo ERREUR : aucun interpreteur Python n'a ete trouve.
echo Installez Python 3.9 ou une version plus recente, puis relancez ce script.
echo.
pause
exit /b 1

:PYTHON_FOUND
set /p "BOOTSTRAP_PY="<"%PYTHON_CANDIDATE_FILE%"
del "%PYTHON_CANDIDATE_FILE%" >nul 2>&1

echo Python detecte :
echo %BOOTSTRAP_PY%
echo.

"%BOOTSTRAP_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)"
if errorlevel 1 (
    echo ERREUR : Python 3.9 minimum est requis.
    echo.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%" (
    echo Creation de l'environnement virtuel...
    "%BOOTSTRAP_PY%" -m venv "%VENV_DIR%"
    if errorlevel 1 goto INSTALL_ERROR
) else (
    echo Environnement virtuel deja present :
    echo %VENV_DIR%
    echo.
)

echo Mise a jour de pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto INSTALL_ERROR

echo Installation de SCARLET et de ses dependances...
"%PYTHON_EXE%" -m pip install -e "%SCARLET_DIR%"
if errorlevel 1 goto INSTALL_ERROR

echo Creation du kernel Jupyter SCARLET...
"%PYTHON_EXE%" -m ipykernel install --user --name scarlet --display-name "SCARLET"
if errorlevel 1 (
    echo.
    echo ATTENTION : le kernel Jupyter n'a pas pu etre enregistre.
    echo Le notebook pourra tout de meme etre lance depuis run_scarlet.cmd.
    echo.
)

if not exist "%SCARLET_DIR%\tutorial_sessions" (
    mkdir "%SCARLET_DIR%\tutorial_sessions"
)

echo.
echo Installation terminee.
echo.
echo Lanceurs disponibles :
echo   - run_scarlet.cmd        ^(JupyterLab et notebooks^)
echo   - run_scarlet_viewer.cmd ^(SCARLET viewer^)
echo.
pause
exit /b 0

:INSTALL_ERROR
echo.
echo ERREUR : l'installation a echoue.
echo.
pause
exit /b 1
