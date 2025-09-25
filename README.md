# new-lease-setup-api

how to run
1. Activate virtual environment

python -m venv my_env
my_env\Scripts\Activate.ps1

2. Install dependencies

pip install fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv pydantic[email]

3. Run api: 
uvicorn main:app --reload --host 127.0.0.1 --port 3002