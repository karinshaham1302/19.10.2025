# Machine Learning & FastAPI Final Project

This project implements a complete, production-style machine learning pipeline served via a FastAPI backend.  
It supports training regression models on a predefined dataset and performing predictions through authenticated API calls using JWT tokens.  
A token-based usage system and a Streamlit dashboard are also included.

---

## 1. TL;DR — Short Version

- Upload `private_lessons_data.csv` → Train ML model (Linear, Decision Tree, Random Forest).  
- Schema is fixed and validated automatically.  
- Model is saved with full metadata (features, label, metrics).  
- JWT authentication required.  
- Each API call consumes user tokens.  
- Predictions use the **latest trained model**.  
- Streamlit dashboard displays users + token balances.

---

## 2. How It Works — System Flow

1. User signs up (`/auth/signup`)
2. User logs in → receives JWT token (`/auth/login`)
3. User uploads `private_lessons_data.csv` to train (`/training/train`)
4. Model is trained, evaluated, saved into `/models/`
5. User performs prediction (`/models/predict/{model_name}`)
6. Server loads latest trained model → returns predicted lesson price  
7. Prediction consumes 5 tokens (configurable)

---

## 3. Project Structure

```text
19.10.2025/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── schemas.py
│   ├── auth_service.py
│   ├── model_service.py
│   └── routers/
│       ├── auth.py
│       ├── training.py
│       └── prediction.py
│
├── data/
│   └── private_lessons_data.csv
│
├── models/
│   └── (saved .pkl models)
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md
4. Dataset
This project uses a predefined schema for lesson-pricing prediction:

Column	Type
subject	string
student_level	string
lesson_minutes	int
teacher_experience_years	int
is_online	string
city	string
teacher_age	int
lesson_price (label)	float

Training strictly requires this structure.

5. Setup & Run
Create Virtual Environment
bash
Copy code
python -m venv .venv
Activate:

Windows

bash
Copy code
.venv\Scripts\activate
Mac / Linux

bash
Copy code
source .venv/bin/activate
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Run API Server
bash
Copy code
uvicorn app.main:app --reload
Open documentation:

arduino
Copy code
http://127.0.0.1:8000/docs
6. Authentication (JWT)
Sign Up
json
Copy code
POST /auth/signup
{
  "username": "user1",
  "password": "pass1234"
}
Log In → Get JWT
json
Copy code
POST /auth/login
{
  "access_token": "<TOKEN>",
  "token_type": "bearer"
}
Add the Token in Swagger
Click Authorize → Paste only the token (no “Bearer”).

7. Train a Model
Endpoint
bash
Copy code
POST /training/train
Form-data Fields
file: private_lessons_data.csv

model_name: linear

model_params: {} (or any JSON)

Example Response
json
Copy code
{
  "status": "success",
  "model_info": {
    "model_name": "linear",
    "n_samples": 100,
    "metrics": {
      "r2": 0.96,
      "mae": 5.17,
      "rmse": 7.43
    }
  }
}
8. List All Trained Models
text
Copy code
GET /models
9. Make Prediction
Endpoint
bash
Copy code
POST /models/predict/linear
Request Body Example
json
Copy code
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
10. Streamlit Admin Dashboard
Run:

bash
Copy code
python -m streamlit run tokens_dashboard.py
Dashboard displays:

All users

Token balances

Usage overview

11. Future Improvements
Add model choices: SVR, XGBoost, Gradient Boosting

Add feature-scaling options

Add admin roles & rate limiting

Dockerize entire stack

Build full web UI for uploading datasets and performing predictions

Add Alembic migrations for database evolution

12. Notes
Dataset schema is fixed for this project

JWT token must be refreshed when expired

All saved models are stored under /models/

Logging is stored under logs/ (auto-created)

Copy code
