@echo off
setlocal

python -m venv venv
call venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup finalizado.
echo Agora copie .env.example para .env e rode run_local.bat
pause