# Machine Learning & FastAPI Final Project

A complete production-style machine learning pipeline served via FastAPI.  
Includes data validation, model training, prediction endpoints, JWT authentication,  
a token-based usage system, and a Streamlit admin dashboard.

---

## Overview

This project lets users:

- Train machine-learning models on a predefined dataset  
- Make predictions using trained models  
- Authenticate using JWT tokens  
- Consume tokens for API operations (training, prediction)  
- View user/token status in an admin dashboard  

Fully implemented with **FastAPI**, **scikit-learn**, **SQLite**, and **Streamlit**.

---

## How It Works

1. User signs up (`/auth/signup`)
2. User logs in → receives a JWT token
3. User trains a model on the predefined dataset (`/training/train`)
4. The model is saved with full metadata inside `/models/`
5. The user sends prediction input to `/models/predict/{model_name}`
6. The system loads the latest model and returns a **rounded prediction (2 decimals)**
7. Token balance is updated according to action

---

## Project Structure

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
│   └── (saved .pkl model files)
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md
Dataset Schema
This project uses a predefined dataset with the following structure:

Column	Type
subject	string
student_level	string
lesson_minutes	int
teacher_experience_years	int
is_online	string
city	string
teacher_age	int
lesson_price (label)	float

Location:

bash
Copy code
data/private_lessons_data.csv
Installation
Create virtual environment
bash
Copy code
python -m venv .venv
Activate environment
Windows

bash
Copy code
.venv\Scripts\activate
Mac / Linux

bash
Copy code
source .venv/bin/activate
Install dependencies
bash
Copy code
pip install -r requirements.txt
Run the Server
bash
Copy code
uvicorn app.main:app --reload
Swagger UI:

arduino
Copy code
http://127.0.0.1:8000/docs
Authentication
Sign Up
json
Copy code
{
  "username": "user1",
  "password": "pass1234"
}
Log In (returns JWT)
json
Copy code
{
  "access_token": "<JWT_TOKEN>"
}
In Swagger → Click Authorize and paste only the token
(no need to type "Bearer").

Token System
Action	Cost
Train model	1
Predict	5

Check balance
bash
Copy code
GET /auth/tokens
Add tokens
bash
Copy code
POST /auth/add_tokens
json
Copy code
{
  "amount": 20
}
Training a Model
Endpoint:

bash
Copy code
POST /training/train
Request body:

json
Copy code
{
  "model_name": "linear",
  "model_params": {}
}
Metrics returned (rounded to 2 decimals):

R²

MAE

MSE

RMSE

Model is saved into /models/.

Making a Prediction
Endpoint:

bash
Copy code
POST /models/predict/linear
Request example:

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
Response example:

json
Copy code
{
  "model_name": "linear",
  "model_id": 2,
  "prediction": 163.04
}
Prediction is always returned with two decimal places.

Streamlit Dashboard
Run:

bash
Copy code
python -m streamlit run tokens_dashboard.py
Dashboard shows:

All users

Token balance per user

Future Improvements
Add more ML models (XGBoost, SVM, Gradient Boosting)

Batch prediction endpoint

Admin vs user roles

Automatic detection of features/labels

Docker deployment

Email alerts for low token balance

Full web UI for training & prediction

Notes
Notebook included: project_info.ipynb

Dataset stored in: data/private_lessons_data.csv

Model predictions depend on the fixed schema

Authentication uses JWT for all protected routes

yaml
Copy code
