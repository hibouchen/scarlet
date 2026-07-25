@echo off
setlocal EnableExtensions

rem ==================================================
rem Configuration
rem ==================================================

set "SCARLET_DIR=C:\Users\gac-sansllb\Document\SCARLET"
set "VENV_DIR=%SCARLET_DIR%\scarlet_venv"

rem Notebook original utilise comme modele
set "TEMPLATE_NOTEBOOK=%SCARLET_DIR%\scarlet\notebooks\tutorial.ipynb"

rem Dossier contenant les notebooks des utilisateurs
set "SESSIONS_DIR=%SCARLET_DIR%\tutorial_sessions"

title Tutoriel SCARLET

rem ==================================================
rem Verifications
rem ==================================================

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo.
    echo ERREUR : environnement virtuel introuvable.
    echo.
    echo Chemin recherche :
    echo %VENV_DIR%
    echo.
    pause
    exit /b 1
)

if not exist "%TEMPLATE_NOTEBOOK%" (
    echo.
    echo ERREUR : notebook modele introuvable.
    echo.
    echo Chemin recherche :
    echo %TEMPLATE_NOTEBOOK%
    echo.
    pause
    exit /b 1
)

rem Verifier que JupyterLab est installe
"%VENV_DIR%\Scripts\python.exe" -c "import jupyterlab" >nul 2>&1

if errorlevel 1 (
    echo.
    echo ERREUR : JupyterLab n'est pas installe.
    echo.
    echo Pour l'installer :
    echo "%VENV_DIR%\Scripts\python.exe" -m pip install jupyterlab
    echo.
    pause
    exit /b 1
)

rem Creer le dossier des sessions si necessaire
if not exist "%SESSIONS_DIR%" (
    mkdir "%SESSIONS_DIR%"
)

rem ==================================================
rem Menu principal
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

if "%CHOIX%"=="1" goto NEW_NOTEBOOK
if "%CHOIX%"=="2" goto OPEN_NOTEBOOK
if "%CHOIX%"=="3" goto FIN

echo.
echo Choix invalide.
pause
goto MENU

rem ==================================================
rem Creer un nouveau notebook
rem ==================================================

:NEW_NOTEBOOK
cls
echo.
echo ==================================================
echo        Creation d'un nouveau notebook
echo ==================================================
echo.
echo Entrez votre nom ou vos initiales.
echo Utilisez de preference uniquement des lettres
echo et des chiffres.
echo.

set "USER_NAME="
set /p "USER_NAME=Nom ou initiales : "

if not defined USER_NAME (
    set "USER_NAME=utilisateur"
)

rem Remplacer les espaces par des tirets bas
set "USER_NAME=%USER_NAME: =_%"

rem Creer un horodatage
for /f %%I in (
    'powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"'
) do set "TIMESTAMP=%%I"

rem Dossier personnel
set "USER_DIR=%SESSIONS_DIR%\%USER_NAME%"

if not exist "%USER_DIR%" (
    mkdir "%USER_DIR%"

    if errorlevel 1 (
        echo.
        echo ERREUR : impossible de creer le dossier :
        echo %USER_DIR%
        echo.
        pause
        goto MENU
    )
)

rem Nom de la nouvelle copie
set "TARGET_NOTEBOOK=%USER_DIR%\tutorial_%USER_NAME%_%TIMESTAMP%.ipynb"

copy "%TEMPLATE_NOTEBOOK%" "%TARGET_NOTEBOOK%" >nul

if errorlevel 1 (
    echo.
    echo ERREUR : impossible de copier le notebook.
    echo.
    pause
    goto MENU
)

echo.
echo Une nouvelle copie a ete creee :
echo.
echo %TARGET_NOTEBOOK%
echo.

goto LAUNCH_JUPYTER

rem ==================================================
rem Selectionner un notebook existant
rem ==================================================

:OPEN_NOTEBOOK
cls
echo.
echo Selectionnez le notebook a ouvrir.
echo.

set "TARGET_NOTEBOOK="

for /f "usebackq delims=" %%I in (`powershell -NoProfile -STA -Command ^
    "Add-Type -AssemblyName System.Windows.Forms; ^
    $dialog = New-Object System.Windows.Forms.OpenFileDialog; ^
    $dialog.Title = 'Selectionner un notebook SCARLET'; ^
    $dialog.InitialDirectory = '%SESSIONS_DIR%'; ^
    $dialog.Filter = 'Notebooks Jupyter (*.ipynb)|*.ipynb|Tous les fichiers (*.*)|*.*'; ^
    $dialog.Multiselect = $false; ^
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { ^
        Write-Output $dialog.FileName ^
    }"`) do set "TARGET_NOTEBOOK=%%I"

rem L'utilisateur a ferme la boite de dialogue
if not defined TARGET_NOTEBOOK (
    goto MENU
)

if not exist "%TARGET_NOTEBOOK%" (
    echo.
    echo ERREUR : le notebook selectionne est introuvable.
    echo.
    echo %TARGET_NOTEBOOK%
    echo.
    pause
    goto MENU
)

goto LAUNCH_JUPYTER

rem ==================================================
rem Lancer JupyterLab
rem ==================================================

:LAUNCH_JUPYTER
cls
echo.
echo ==================================================
echo              Lancement de JupyterLab
echo ==================================================
echo.
echo Notebook ouvert :
echo.
echo %TARGET_NOTEBOOK%
echo.

rem Se placer dans le dossier contenant le notebook
for %%I in ("%TARGET_NOTEBOOK%") do set "NOTEBOOK_DIR=%%~dpI"

cd /d "%NOTEBOOK_DIR%"

"%VENV_DIR%\Scripts\python.exe" -m jupyter lab "%TARGET_NOTEBOOK%"

if errorlevel 1 (
    echo.
    echo ==================================================
    echo ERREUR : JupyterLab n'a pas pu etre lance
    echo ==================================================
    echo.
    pause
)

goto MENU

rem ==================================================
rem Fermeture
rem ==================================================

:FIN
endlocal
exit /b 0