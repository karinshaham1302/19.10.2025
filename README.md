📄 README — Machine Learning & FastAPI Final Project

This project implements an end-to-end machine-learning system using FastAPI, including dataset upload, model training, prediction, authentication, token management, and a Streamlit dashboard for monitoring usage.

1. Project Structure
19.10.2025/
│
├── app/
│   ├── routers/
│   │   ├── auth.py
│   │   ├── training.py
│   │   └── prediction.py
│   │
│   ├── models/                  # Trained ML models (.pkl)
│   ├── logs/                    # Log files
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

2.2 Activate environment (Windows)
.venv\Scripts\activate

2.3 Install dependencies
pip install -r requirements.txt

3. Run the FastAPI Server
uvicorn app.main:app --reload


Swagger UI:

http://127.0.0.1:8000/docs

4. Authentication (JWT)
4.1 Sign up
{
  "username": "user1",
  "password": "pass1234"
}

4.2 Log in

Returns JWT:

{
  "access_token": "<JWT_TOKEN>"
}

4.3 Authorizing in Swagger

Press Authorize and paste only the token (without “Bearer”).

5. Token System

Model training → cost: 1 token

Model prediction → cost: 5 tokens

5.1 Check tokens
GET /auth/tokens

5.2 Add tokens
{
  "amount": 20
}

6. Training a Model

Endpoint:

POST /training/train


Parameters:

file (CSV)

model_name (linear, decision_tree, random_forest)

model_params (optional JSON)

All metrics (r², MAE, MSE, RMSE) are stored and rounded to two decimal places.

7. Prediction

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

8. System Flow (How It Works)
User → Login → JWT
          │
          ▼
   Authorize in Swagger
          │
          ▼
 Upload CSV → Train Model → Save .pkl
          │
          ▼
 Make Prediction (tokens deducted)
          │
          ▼
 Streamlit dashboard for monitoring

9. Streamlit Dashboard

Run:

python -m streamlit run tokens_dashboard.py


Displays:

Users

Token balances

10. Future Improvements

Add advanced ML algorithms (XGBoost, SVM, Gradient Boosting)

Add batch prediction endpoint

Add admin role + permission tiers

Add frontend UI

Add Docker deployment

Automatic feature/label detection

11. Notes

Dataset used: private_lessons_data.csv

Schema is predefined to ensure consistent predictions

JWT secures access to protected endpoints

All results are rounded to two decimal places

✔ גרסה מקוצרת (Summary)
FastAPI ML server:
- CSV upload → train model
- Predict using saved models
- JWT login
- Token-based usage control
- Streamlit monitoring dashboard
