@echo off
setlocal

rem ==================================================
rem Configuration des chemins
rem ==================================================

set "SCARLET_DIR=C:\Users\gac-sansllb\Documents\SCARLET"
set "VENV_DIR=%SCARLET_DIR%\scarlet_venv"
set "NOTEBOOK=%SCARLET_DIR%\scarlet\notebooks\tutorial.ipynb"

rem ==================================================
rem Vérification de l'environnement virtuel
rem ==================================================

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    cls
    echo.
    echo ==================================================
    echo ERREUR : environnement virtuel introuvable
    echo ==================================================
    echo.
    echo Chemin recherche :
    echo %VENV_DIR%
    echo.
    echo Verifiez que le chemin est correct.
    echo Verifiez notamment si le dossier Windows se nomme
    echo "Documents".
    echo.
    pause
    exit /b 1
)

rem ==================================================
rem Préparation de l'environnement
rem ==================================================

cd /d "%SCARLET_DIR%"

call "%VENV_DIR%\Scripts\activate.bat"

title SCARLET

rem ==================================================
rem Menu principal
rem ==================================================

:MENU
cls
echo.
echo ==================================================
echo                    SCARLET
echo ==================================================
echo.
echo   1 - Lancer SCARLET Viewer
echo.
echo   2 - Ouvrir tutorial.ipynb dans JupyterLab
echo.
echo   3 - Quitter
echo.
echo ==================================================
echo.

set /p "CHOIX=Votre choix : "

if "%CHOIX%"=="1" goto VIEWER
if "%CHOIX%"=="2" goto JUPYTERLAB
if "%CHOIX%"=="3" goto FIN

echo.
echo Choix invalide. Entrez 1, 2 ou 3.
echo.
pause
goto MENU

rem ==================================================
rem Lancement de SCARLET Viewer
rem ==================================================

:VIEWER
cls
echo.
echo ==================================================
echo Lancement de SCARLET Viewer
echo ==================================================
echo.
echo Dossier de travail :
echo %SCARLET_DIR%
echo.

scarlet viewer

if errorlevel 1 (
    echo.
    echo ==================================================
    echo ERREUR
    echo ==================================================
    echo.
    echo SCARLET Viewer n'a pas pu etre lance.
    echo.
    echo Verifiez que SCARLET est correctement installe
    echo dans l'environnement virtuel :
    echo.
    echo %VENV_DIR%
    echo.
    pause
)

goto MENU

rem ==================================================
rem Lancement de JupyterLab
rem ==================================================

:JUPYTERLAB
cls
echo.
echo ==================================================
echo Lancement de JupyterLab
echo ==================================================
echo.
echo Notebook :
echo %NOTEBOOK%
echo.

if not exist "%NOTEBOOK%" (
    echo ==================================================
    echo ERREUR : notebook introuvable
    echo ==================================================
    echo.
    echo Chemin recherche :
    echo %NOTEBOOK%
    echo.
    pause
    goto MENU
)

if not exist "%VENV_DIR%\Scripts\jupyter-lab.exe" (
    echo ==================================================
    echo ERREUR : JupyterLab n'est pas installe
    echo ==================================================
    echo.
    echo Pour installer JupyterLab, executez :
    echo.
    echo "%VENV_DIR%\Scripts\python.exe" -m pip install jupyterlab
    echo.
    pause
    goto MENU
)

rem Lancer JupyterLab dans une nouvelle fenêtre
start "JupyterLab SCARLET" "%VENV_DIR%\Scripts\python.exe" -m jupyter lab "%NOTEBOOK%"

echo.
echo JupyterLab a ete lance.
echo Le notebook devrait s'ouvrir dans le navigateur.
echo.
pause
goto MENU

rem ==================================================
rem Fermeture
rem ==================================================

:FIN
endlocal
exit