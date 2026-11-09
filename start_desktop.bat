@echo off
title PS1 3D Reconstruction Studio
echo Starting PS1 3D Reconstruction Desktop Application...
python desktop_app.py
if errorlevel 1 (
    echo.
    echo An error occurred while running the application.
    pause
)

