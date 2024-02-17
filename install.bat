
@echo off
setlocal
echo Starting InstantID CPU env installation...

set "PYTHON_COMMAND=python"

call python --version > nul 2>&1
if %errorlevel% equ 0 (
    echo Python command check :OK
) else (
    echo "Error: Python command not found,please install Python(Recommended : Python 3.10 or higher) and try again."
    pause
    exit /b 1
    
)

:check_python_version
for /f "tokens=2" %%I in ('%PYTHON_COMMAND% --version 2^>^&1') do (
    set "python_version=%%I"
)

echo Python version: %python_version%

%PYTHON_COMMAND% -m venv "%~dp0env" 
call "%~dp0env\Scripts\activate.bat" && pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu 
call "%~dp0env\Scripts\activate.bat" && pip install -r "%~dp0requirements.txt"
echo InstantID CPU env installation completed.
pause