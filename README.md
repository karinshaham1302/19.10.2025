# Machine Learning & FastAPI Final Project

This project implements an end-to-end machine learning system for predicting private lesson prices using FastAPI, scikit-learn, JWT authentication, and a token-based usage system.  
A Streamlit admin dashboard is also included.

---

## 1. TL;DR (Short Summary)

- Fixed dataset (`private_lessons_data.csv`)
- Train ML models (Linear Regression, Decision Tree, Random Forest)
- Make predictions through API calls
- JWT authentication + user token system
- Models saved as `.pkl` with metadata
- Streamlit dashboard shows users & tokens

---

## 2. How It Works (System Flow)

1. User signs up → `/auth/signup`  
2. User logs in → receives JWT → `/auth/login`  
3. Upload CSV to train model → `/training/train`  
4. Model is trained, evaluated, and saved  
5. User requests prediction → `/models/predict/{model_name}`  
6. Tokens deducted for actions  
7. Admin views tokens in Streamlit dashboard  

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
│   └── (saved .pkl models + metadata JSON)
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md
4. Dataset and Notebook
4.1 Dataset: private_lessons_data.csv
Used for both EDA and model training via API.

Main columns:

subject

student_level

lesson_minutes

teacher_experience_years

is_online

city

teacher_age

lesson_price (label)

Feature columns used by the ML pipeline:

python
Copy code
["subject", "student_level", "lesson_minutes",
 "teacher_experience_years", "is_online", "city"]
Label column:

python
Copy code
"lesson_price"
4.2 Notebook: project_info.ipynb
Contains:

Dataset loading

EDA (seaborn histograms, distributions, correlations)

Linear Regression training example

Evaluation (R², MAE, RMSE rounded to 2 decimals)

Summary insights

5. Technologies
Component	Technology
Web API	FastAPI
Auth	JWT (python-jose)
ML	scikit-learn
Data	pandas
Storage	joblib + SQLite
Dashboard	Streamlit
Python Version	3.x

6. Installation
6.1 Clone the repository
bash
Copy code
git clone https://github.com/karinshaham1302/19.10.2025.git
cd 19.10.2025
6.2 Create virtual environment
Windows:

bash
Copy code
python -m venv .venv
.venv\Scripts\activate
macOS/Linux:

bash
Copy code
python -m venv .venv
source .venv/bin/activate
6.3 Install dependencies
bash
Copy code
pip install -r requirements.txt
7. Run the FastAPI Server
bash
Copy code
uvicorn app.main:app --reload
Open:

API root → http://127.0.0.1:8000/

Swagger UI → http://127.0.0.1:8000/docs

8. Authentication & Tokens
8.1 Token Costs
Action	Tokens
Train model	1
Train multiple	1
Predict	5

8.2 Auth Flow
Sign up → /auth/signup

Log in → /auth/login

Copy the access_token

In Swagger → click Authorize → paste only the token

All protected endpoints now work

8.3 Main Auth Endpoints
Method	Endpoint	Description
POST	/auth/signup	Create new user
POST	/auth/login	Get JWT
GET	/auth/tokens	Check token balance
POST	/auth/add_tokens	Add tokens
DELETE	/auth/remove_user	Delete user

9. Model Service (ML Logic)
Located in: app/model_service.py

Provides:

Dataset validation

Preprocessing (OneHotEncoder + numeric passthrough)

Model creation

Training (Linear / Decision Tree / Random Forest)

Metrics (R², MAE, MSE, RMSE — all rounded to 2 decimals)

Saving models to disk

Saving metadata JSON

Also includes:

get_all_models()

get_latest_model_record()

load_model_from_record()

10. Training API
10.1 Train Single Model
POST /training/train

Fields:

file — CSV (e.g., private_lessons_data.csv)

model_name — "linear", "decision_tree", "random_forest"

model_params — optional JSON string

Requires JWT + tokens

Example response:

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
11. Prediction API
11.1 List Models
GET /models/

Returns model metadata + metrics.

11.2 Predict
POST /models/predict/{model_name}

Request body:

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
Response:

json
Copy code
{
  "model_name": "linear",
  "prediction": 163.04
}
12. Streamlit Dashboard
Run:

bash
Copy code
python -m streamlit run tokens_dashboard.py
Dashboard shows:

All users

Token balances

System statistics

13. Testing the API (Swagger)
Recommended flow:

/auth/signup

/auth/login

Authorize (paste token)

/auth/add_tokens

/training/train

/models/

/models/predict/linear

14. Future Improvements
Support more ML models (XGBoost, SVR, Gradient Boosting)

Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

Add full model comparison dashboard

Improve error messages

Add training history per user

Add charts to Streamlit dashboard

Add automated tests (pytest)

Package with Docker for deployment
