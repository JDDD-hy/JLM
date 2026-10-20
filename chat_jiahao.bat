@echo off
chcp 65001 >nul
title Jiahao Language Model
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore
set TRANSFORMERS_VERBOSITY=error
set TOKENIZERS_PARALLELISM=false
echo Loading model (about 30 seconds)...
.venv\Scripts\python.exe -m src.app %*
echo.
pause
