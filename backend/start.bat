@echo off
echo Starting SEFS Backend...
echo.
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
