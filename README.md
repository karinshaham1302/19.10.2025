Machine Learning & FastAPI Final Project

This project implements an end-to-end machine learning system for predicting private lesson prices using FastAPI, scikit-learn, JWT authentication, token usage tracking, and a Streamlit dashboard.

1. TL;DR (Short Summary)

Dataset: data/private_lessons_data.csv

Train models: Linear Regression, Decision Tree, Random Forest

Predictions are made through authenticated API calls

JWT authentication + token balance per user

Models saved as .pkl with metadata and metrics

Streamlit dashboard shows users and tokens

2. How It Works (System Flow)

User signs up → /auth/signup

User logs in → receives JWT token

User uploads dataset to train a model

Model is trained, evaluated, saved under /models/

User sends prediction request

Token balance decreases according to action

Admin views info in Streamlit dashboard

3. Project Structure
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
│   └── (saved .pkl models + models_metadata.json)
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md

4. Dataset and Notebook
4.1 Dataset: private_lessons_data.csv

Main columns:

Column	Type
subject	string
student_level	string
lesson_minutes	int
teacher_experience_years	int
is_online	string
city	string
teacher_age	int
lesson_price (label)	float

Feature columns:

["subject", "student_level", "lesson_minutes",
 "teacher_experience_years", "is_online", "city"]


Target:

"lesson_price"

4.2 Notebook: project_info.ipynb

Includes:

EDA: distributions, correlations, summary

Simple Linear Regression training example

Metrics: R², MAE, RMSE (rounded to 2 decimals)

Visualizations using matplotlib/seaborn

5. Technologies
Component	Technology
API	FastAPI
Auth	JWT (python-jose)
Password Hashing	bcrypt
ML	scikit-learn
Data	pandas
Storage	joblib + SQLite
Dashboard	Streamlit
Python	3.x
6. Installation
6.1 Clone
git clone https://github.com/karinshaham1302/19.10.2025.git
cd 19.10.2025

6.2 Virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate


macOS/Linux:

python -m venv .venv
source .venv/bin/activate

6.3 Install dependencies
pip install -r requirements.txt

7. Run FastAPI
uvicorn app.main:app --reload


Open:

API → http://127.0.0.1:8000/

Swagger UI → http://127.0.0.1:8000/docs

8. Authentication & Tokens
8.1 Token Costs
Action	Cost
Train model	1
Train multiple	1
Predict	5
8.2 Flow

Sign up → /auth/signup

Login → /auth/login

Copy access_token

In Swagger → “Authorize” → paste token (without Bearer)

8.3 Endpoints
Method	Endpoint	Description
POST	/auth/signup	Create user
POST	/auth/login	Get JWT
GET	/auth/tokens	Check balance
POST	/auth/add_tokens	Add tokens
DELETE	/auth/remove_user	Delete user
9. Model Service (ML Logic)

Located in app/model_service.py.

Handles:

Dataset validation

Preprocessing (OneHotEncoder + numeric passthrough)

Model creation:

LinearRegression

DecisionTreeRegressor

RandomForestRegressor

Metrics: R², MAE, MSE, RMSE

Saving model + metadata

Utility functions:

get_all_models()

get_latest_model_record()

load_model_from_record()

10. Training API
10.1 Train single model
POST /training/train


Fields:

file (CSV)

model_name

model_params (optional JSON)

Response example:

{
  "model_name": "linear",
  "metrics": {
    "r2": 0.96,
    "mae": 5.17,
    "mse": 52.83,
    "rmse": 7.27
  }
}

10.2 Train multiple models
POST /training/train_multi


Returns metrics for each model and best model.

11. Prediction API
11.1 List models
GET /models/

11.2 Predict
POST /models/predict/{model_name}


Request:

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


Response:

{
  "model_name": "linear",
  "prediction": 163.04
}

12. Streamlit Dashboard

Run:

python -m streamlit run tokens_dashboard.py


Shows:

All users

Token balances

System totals

13. Testing API (Swagger)

Recommended flow:

/auth/signup

/auth/login

Authorize

/auth/add_tokens

/training/train

/models/

/models/predict/{model_name}

14. Future Improvements

More model types (XGBoost, SVR, Gradient Boosting)

Hyperparameter tuning

Better error messages

Model history per user

Streamlit charts

Add pytest tests

Docker
