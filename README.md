# new-lease-setup-api

How to setup 

1. Create .env file
   DATABASE_URL=postgresql://username:password@localhost:5432/lease_request_poc

2. Activate virtual environment

3. Install dependencies

    pip install fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv pydantic[email]

4. Run api: 
    uvicorn main:app --reload --host 127.0.0.1 --port 3002

Note:
The postgresql tables will be automatically created
