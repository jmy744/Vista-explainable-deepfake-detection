@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: creating a Python 3.12 environment...
  py -3.12 -m venv .venv
  if errorlevel 1 goto fail
 )
if not exist ".venv\demo-ready.txt" (
  .venv\Scripts\python -m pip install -r requirements.txt
  if errorlevel 1 goto fail
  echo ready > .venv\demo-ready.txt
)
.venv\Scripts\python -m streamlit run app.py
pause
exit /b
:fail
echo Setup did not finish. Check that Python 3.12 is installed and you have internet access.
echo Use the commands in README.md to complete setup before running again.
pause
