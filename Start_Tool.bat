@echo off
title k-NN Word Pair Semantic Classifier
chcp 65001 >nul
cls

echo ======================================================================
echo    k-NN Word Pair Classifier - Machine Learning Desktop App
echo ======================================================================
echo.

:: Kiem tra Python
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set PY_CMD=py
    goto RUN
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PY_CMD=python
    goto RUN
)

echo [ERROR] Khong tim thay Python! Vui long cai dat Python 3.9 tro len.
pause
exit /b 1

:RUN
echo [1/2] Dang kiem tra mo hinh...
if not exist "models\knn_pipeline.pkl" (
    echo [INFO] Dang huan luyen mo hinh lan dau...
    %PY_CMD% train_and_test.py
)

echo [2/2] Dang khoi dong ung dung Desktop App...
start "" %PY_CMD% app_gui.py
exit /b 0
