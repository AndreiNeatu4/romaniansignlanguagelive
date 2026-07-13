@echo off
REM ============================================================================
REM  Romanian Sign Language - antrenare + export + build, intr-un singur script.
REM  Ruleaza pe CPU (nu necesita placa video dedicata).
REM
REM  Folosire:
REM    run_all.bat                          -> ruleaza tot (extragere+antrenare+export+build)
REM    run_all.bat --skip-extract --skip-images   -> sare peste extragere (daca data/ exista deja)
REM ============================================================================

chcp 65001 >nul
cd /d "%~dp0"

REM Foloseste interpretorul din venv (.venv, Python 3.11) daca exista; altfel
REM cade pe "python" din PATH. Venv-ul e necesar fiindca mediapipe nu merge pe 3.8.
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
echo Folosesc interpretorul: %PY%

echo.
echo ================================================================
echo  PAS 0/3 - Verificare pachete Python necesare
echo ================================================================
"%PY%" -c "import importlib.util as u, sys; mods=['torch','cv2','mediapipe','numpy','albumentations','sklearn','matplotlib','seaborn','tqdm','onnx']; missing=[m for m in mods if u.find_spec(m) is None]; (print('Toate pachetele necesare sunt prezente.') if not missing else print('LIPSESC pachete: '+', '.join(missing)+'\n  Instaleaza-le cu:\n  pip install opencv-python mediapipe albumentations scikit-learn matplotlib seaborn tqdm onnx')); sys.exit(1 if missing else 0)"
if errorlevel 1 goto :error_deps

echo.
echo ================================================================
echo  PAS 1/3 - Pipeline de antrenare (extragere, pregatire, antrenare)
echo ================================================================
"%PY%" run_pipeline.py %*
if errorlevel 1 goto :error

echo.
echo ================================================================
echo  PAS 2/3 - Export model in format ONNX
echo ================================================================
"%PY%" static-site\tools\export_onnx.py
if errorlevel 1 goto :error

echo.
echo ================================================================
echo  PAS 3/3 - Construire asset-uri pentru site (css + videoclipuri + videos.json)
echo ================================================================
"%PY%" static-site\tools\build_assets.py
if errorlevel 1 goto :error

echo.
echo ================================================================
echo  GATA! Model antrenat, exportat in ONNX si asset-uri reconstruite.
echo  Site-ul actualizat se afla in folderul:  static-site\
echo  Il poti testa local cu:  cd static-site ^&^& python -m http.server 8080
echo ================================================================
pause
exit /b 0

:error_deps
echo.
echo [OPRIT] Lipsesc pachete Python. Instaleaza-le (vezi comanda de mai sus) si reruleaza.
pause
exit /b 1

:error
echo.
echo [EROARE] Un pas a esuat (cod %errorlevel%). Vezi mesajele de mai sus.
pause
exit /b 1
