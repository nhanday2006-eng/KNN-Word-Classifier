#!/bin/bash
# Script khoi dong k-NN Word Pair Classifier tren Linux/macOS

cd "$(dirname "$0")"

echo "======================================================================"
echo "   k-NN Word Pair Classifier - Machine Learning Desktop App"
echo "======================================================================"
echo ""

if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python 3 khong tim thay! Vui long cai dat Python 3.9+."
    exit 1
fi

if [ ! -f "models/knn_pipeline.pkl" ]; then
    echo "[INFO] Dang huan luyen mo hinh lan dau..."
    $PY_CMD train_and_test.py
fi

echo "[INFO] Dang khoi dong ung dung Desktop App..."
$PY_CMD app_gui.py
