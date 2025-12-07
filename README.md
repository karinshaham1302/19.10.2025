# Machine Learning & FastAPI Final Project

This project implements a complete machine-learning workflow using FastAPI.  
It includes dataset handling, model training, prediction endpoints, JWT authentication,  
token-based usage control, and an admin Streamlit dashboard.

---

## 1. TL;DR — Short Version
- Train ML models using a predefined dataset (`private_lessons_data.csv`)
- Supported models: **Linear Regression**, **Decision Tree**, **Random Forest**
- Schema is fixed and validated automatically
- JWT Authentication is required
- API actions consume user tokens (train = 1, predict = 5)
- Predictions use the **latest trained model**
- Streamlit dashboard shows all users and their tokens

---

## 2. How It Works — System Flow
1. User signs up (`/auth/signup`)
2. User logs in and receives a **JWT token**
3. User uploads dataset and trains a model (`/training/train`)
4. Model is saved into `/models/` with metadata
5. User sends input values to `/models/predict/{model_name}`
6. API loads the latest trained model and returns a prediction
7. Token balance is updated accordingly

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
│   └── (saved .pkl model files)
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md
4. Dataset Schema
The project uses this fixed dataset structure:

Column	Type
subject	string
student_level	string
lesson_minutes	int
teacher_experience_years	int
is_online	string
city	string
teacher_age	int
lesson_price (label)	float

Training strictly requires this schema.

Dataset file location:

bash
Copy code
data/private_lessons_data.csv
5. Environment Setup
5.1 Create virtual environment
bash
Copy code
python -m venv .venv
5.2 Activate environment
Windows:

bash
Copy code
.venv\Scripts\activate
Mac / Linux:

bash
Copy code
source .venv/bin/activate
5.3 Install requirements
bash
Copy code
pip install -r requirements.txt
6. Run the FastAPI Server
bash
Copy code
uvicorn app.main:app --reload
Swagger UI will be available at:

arduino
Copy code
http://127.0.0.1:8000/docs
7. Authentication Workflow
7.1 Sign Up
json
Copy code
{
  "username": "user1",
  "password": "pass1234"
}
7.2 Log In (returns JWT)
json
Copy code
{
  "access_token": "<JWT_TOKEN>"
}
7.3 Use JWT in Swagger
Click Authorize → paste only the token
(no need to type "Bearer").

8. Token System
Action	Cost
Train model	1
Predict	5

8.1 Check token balance
bash
Copy code
GET /auth/tokens
8.2 Add tokens
bash
Copy code
POST /auth/add_tokens
Body example:

json
Copy code
{
  "amount": 20
}
9. Train a Model
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
Response returns:

metrics (R², MAE, MSE, RMSE — all rounded to 2 decimals)

model metadata

model path

10. Make a Prediction
Endpoint:

bash
Copy code
POST /models/predict/linear
Example request:

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
Example response:

json
Copy code
{
  "model_name": "linear",
  "model_id": 2,
  "prediction": 163.04
}
Prediction is always returned rounded to two decimal places.

11. Streamlit Admin Dashboard
Run:

bash
Copy code
python -m streamlit run tokens_dashboard.py
Dashboard displays:

All users

Remaining tokens per user

12. Future Improvements
Add advanced ML algorithms (XGBoost, SVM, Gradient Boosting)

Batch prediction endpoint

Admin role vs. user role

Automatic inference of feature/label columns

Docker support

Email notifications for low token balance

Web UI for model training and prediction

13. Notes
Jupyter notebook for exploration: project_info.ipynb

Dataset location: data/private_lessons_data.csv

All predictions follow the predefined schema

Authentication uses JWT for secure access

