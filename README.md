📄 README — Machine Learning & FastAPI Final Project

This project implements a full machine-learning pipeline using FastAPI.
It includes dataset upload, model training, prediction, JWT authentication, token management, and a Streamlit dashboard for user monitoring.

1. Project Structure
19.10.2025/
│
├── app/
│   ├── routers/
│   │   ├── auth.py
│   │   ├── training.py
│   │   └── prediction.py
│   │
│   ├── models/                  # Stored trained ML models (.pkl)
│   ├── logs/                    # Application logs
│   │
│   ├── __init__.py
│   ├── auth_service.py
│   ├── model_service.py
│   ├── database.py
│   ├── schemas.py
│   ├── config.py
│   └── main.py
│
├── data/
│   └── private_lessons_data.csv
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md

2. Environment Setup
2.1 Create virtual environment
python -m venv .venv

2.2 Activate virtual environment (Windows)
.venv\Scripts\activate

2.3 Install dependencies
pip install -r requirements.txt

3. Running the FastAPI Server
uvicorn app.main:app --reload


Open Swagger UI:

http://127.0.0.1:8000/docs

4. Authentication (JWT)
4.1 Sign up
{
  "username": "user1",
  "password": "pass1234"
}

4.2 Login

Returns a JWT token.

{
  "access_token": "<JWT_TOKEN>"
}

4.3 Using JWT in Swagger

Click Authorize and paste the token without the word Bearer.

5. Token System

Training a model = 1 token

Prediction = 5 tokens

5.1 Check tokens
GET /auth/tokens

5.2 Add tokens
{
  "amount": 20
}

6. Training a Model

Endpoint:

POST /training/train


Form-data:

file: dataset (.csv)

model_name: linear / decision_tree / random_forest

model_params: optional JSON

The server outputs metrics:

r2

mae

mse

rmse (added)

All rounded to two decimal places.

7. Making Predictions

Endpoint:

POST /models/predict/{model_name}


Example request:

{
  "data": {
    "subject": "math",
    "student_level": "high_school",
    "lesson_minutes": 60,
    "teacher_experience_years": 5,
    "is_online": "yes",
    "city": "Tel Aviv"
  }
}


Example response:

{
  "model_name": "linear",
  "model_id": 2,
  "prediction": 163.04
}

8. How It Works — System Flow
User → Signup/Login → Receives JWT
              │
              ▼
    Authorize in Swagger
              │
              ▼
    Upload CSV → Train Model → Model saved (.pkl)
              │
              ▼
     Make Prediction → Tokens deducted
              │
              ▼
   View usage → Streamlit dashboard

9. Streamlit Dashboard

Run dashboard:

python -m streamlit run tokens_dashboard.py


Displays:

List of users

Tokens per user

10. Future Improvements

Add additional ML models (XGBoost, SVM, Gradient Boosting)

Add batch prediction endpoint

Add role-based access (Admin / User)

Add full frontend UI

Add Docker deployment

Automate feature/label detection from datasets

11. Notes

Dataset used: private_lessons_data.csv

Schema is predefined → prevents invalid predictions

JWT secures protected endpoints

Metrics are standardized across models

🔹 גרסה מקוצרת (Summary)
FastAPI ML server:
- Train models from CSV
- Predict using latest model
- JWT login system
- Token-based usage control
- Streamlit dashboard for monitoring

