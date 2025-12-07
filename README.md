# Machine Learning & FastAPI Final Project

This project implements a complete machine learning workflow served through FastAPI:  
dataset upload, model training, prediction, JWT authentication, token-based usage system,  
and a Streamlit dashboard for monitoring users and tokens.

---

## 1. TL;DR – Short Summary

- Fixed dataset: **private_lessons_data.csv**
- Models: Linear Regression, Decision Tree, Random Forest  
- Training & prediction via FastAPI  
- JWT authentication (signup/login)  
- Each action consumes **tokens**  
- Models saved with full metadata  
- Streamlit dashboard for user/token monitoring  
- Full EDA in project_info.ipynb  

---

## 2. How It Works – System Flow

1. User signs up (`/auth/signup`)
2. User logs in → receives JWT token  
3. User uploads dataset → trains a model (`/training/train`)
4. Model is trained, evaluated, and saved  
5. User requests prediction (`/models/predict/{model_name}`)
6. System loads latest trained model  
7. Tokens are deducted for each action  

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
│   └── (trained .pkl models)
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md
4. Dataset Description
The project uses a fixed CSV file containing private lesson data.

Column	Description
subject	Lesson subject
student_level	Student level
lesson_minutes	Lesson duration
teacher_experience_years	Years of experience
is_online	yes/no
city	City name
teacher_age	Age of teacher
lesson_price	Target label

EDA and visualizations appear in project_info.ipynb.

5. Environment Setup
Create virtual environment
Windows

bash
Copy code
python -m venv .venv
.venv\Scripts\activate
Mac/Linux

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
Install dependencies
bash
Copy code
pip install -r requirements.txt
6. Running the FastAPI Server
bash
Copy code
uvicorn app.main:app --reload
Access:

Swagger UI → http://127.0.0.1:8000/docs

Health check → http://127.0.0.1:8000/health

7. Authentication (JWT)
Signup
json
Copy code
{
  "username": "user1",
  "password": "pass1234"
}
Login
json
Copy code
{
  "username": "user1",
  "password": "pass1234"
}
Response:

json
Copy code
{
  "access_token": "<JWT>",
  "token_type": "bearer"
}
Authorize in Swagger
Paste only the JWT (without “Bearer”).

8. Token System
Action	Token Cost
Train model	1
Train multiple models	1
Predict	5

Check tokens:

json
Copy code
GET /auth/tokens
Add tokens:

json
Copy code
POST /auth/add_tokens
{
  "amount": 20
}
9. Train a Model – /training/train
Form-data fields:
file – CSV file

model_name – linear / decision_tree / random_forest

model_params – JSON (optional)

Example response (simplified):
json
Copy code
{
  "status": "success",
  "model_info": {
    "model_name": "linear",
    "metrics": {
      "r2": 0.96,
      "mae": 5.17,
      "mse": 52.83,
      "rmse": 7.27
    }
  }
}
10. Train Multiple Models – /training/train_multi
Provides best model by R².
Accepts JSON list of model names.

Example:

json
Copy code
["linear", "random_forest"]
11. List All Models – /models/
Returns metadata of all trained models:

json
Copy code
{
  "models": [
    {
      "model_name": "linear",
      "r2": 0.96
    }
  ]
}
12. Make a Prediction – /models/predict/{model_name}
Body format:
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
  "prediction": 163.04
}
13. Streamlit Dashboard
Run:

bash
Copy code
python -m streamlit run tokens_dashboard.py
Shows:

All users

Their token balance

Total system tokens

14. Jupyter Notebook – EDA & Model Evaluation
project_info.ipynb includes:

Dataset exploration

Distributions

Correlations

Regression model training

Error metrics

Summary of findings

This notebook is part of the final project documentation.

15. Future Improvements
Add more ML models (XGBoost, SVM, Gradient Boosting)

Docker container for easy deployment

Batch predictions API

Email alerts when tokens are low

Frontend UI for uploading CSV & predicting

Fully configurable dataset schema

yaml
Copy code
