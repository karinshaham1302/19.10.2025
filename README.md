# Machine Learning & FastAPI – Final Project

This project implements an end-to-end machine learning system for predicting private-lesson prices using:
FastAPI, scikit-learn, JWT authentication, a token-based usage system, and a Streamlit admin dashboard.

---

## 1. TLDR (Short Summary)

- Fixed dataset: `data/private_lessons_data.csv`
- Train ML models: Linear Regression, Decision Tree, Random Forest
- Make predictions via authenticated API calls
- JWT-based authentication with per-user token balance
- Models saved as `.pkl` files with full metadata and metrics
- Streamlit dashboard shows users and tokens

---

## 2. System Flow

1. User signs up → `POST /auth/signup`
2. User logs in → receives JWT token → `POST /auth/login`
3. User uploads dataset and trains a model → `POST /training/train`
4. Model is trained, evaluated, and saved under `models/` (+ JSON metadata)
5. User sends a prediction request → `POST /models/predict/{model_name}`
6. System loads the latest model and returns the predicted price
7. Each action consumes tokens according to configured costs
8. Admin can view users and tokens via the Streamlit dashboard

---

## 3. Project Structure

```text
19.10.2025/
│
├── app/
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
│   ├── (saved .pkl model files)
│   └── models_metadata.json
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md
4. Dataset
4.1 CSV: private_lessons_data.csv
Column	Type
subject	string
student_level	string
lesson_minutes	int
teacher_experience_years	int
is_online	string
city	string
teacher_age	int
lesson_price (label)	float

Feature columns used for training:

python
Copy code
["subject", "student_level", "lesson_minutes",
 "teacher_experience_years", "is_online", "city"]
Target column:

python
Copy code
"lesson_price"
5. Jupyter Notebook – project_info.ipynb
The notebook includes:

Loading the dataset with pandas

df.head(), df.info(), df.describe()

Exploratory Data Analysis:

price distribution

lesson duration distribution

teacher experience distribution

price by subject / student_level / city / online vs offline

correlation heatmap

Simple Linear Regression training example

Metrics: R², MAE, RMSE (printed with two decimal places)

Short summary of the main insights

The notebook uses the same CSV as the FastAPI endpoints.

6. Technologies
Component	Technology
API	FastAPI
Auth	JWT (python-jose)
Password hashing	bcrypt
ML engine	scikit-learn
Data processing	pandas
Model storage	joblib + JSON
Database	SQLite
Admin dashboard	Streamlit
Python version	3.x

All dependencies are listed in requirements.txt.

7. Installation
7.1 Clone the Repository
bash
Copy code
git clone https://github.com/karinshaham1302/19.10.2025.git
cd 19.10.2025
7.2 Create Virtual Environment
Windows

bash
Copy code
python -m venv .venv
.venv\Scripts\activate
macOS / Linux

bash
Copy code
python -m venv .venv
source .venv/bin/activate
7.3 Install Required Packages
bash
Copy code
pip install -r requirements.txt
8. Running the FastAPI Server
From the project root:

bash
Copy code
uvicorn app.main:app --reload
Open:

API root: http://127.0.0.1:8000/

Swagger UI: http://127.0.0.1:8000/docs

9. Authentication & Tokens
Users and tokens are stored in an SQLite database (for example data/app.db).

9.1 Token Costs
Action	Tokens
Train model	1
Train multiple	1
Predict	5

9.2 Auth Flow
Sign up → POST /auth/signup

Log in → POST /auth/login

Copy the access_token value

In Swagger (/docs), click Authorize and paste only the token (without the word Bearer)

All protected endpoints now accept requests with that JWT

9.3 Auth Endpoints
Method	Path	Description
POST	/auth/signup	Register new user
POST	/auth/login	Login and get JWT token
GET	/auth/tokens	Get current token count
POST	/auth/add_tokens	Add tokens to user
DELETE	/auth/remove_user	Remove user

10. Model Service (Machine Learning Logic)
Location: app/model_service.py

Responsibilities:

Validate dataset structure

Build preprocessing using ColumnTransformer + OneHotEncoder

Create models:

LinearRegression

DecisionTreeRegressor

RandomForestRegressor

Train models with train/test split

Compute metrics:

R²

MAE

MSE

RMSE

Round metrics to two decimal places

Save the fitted pipeline as .pkl under models/

Save metadata in models_metadata.json

Utility functions:

train_model(...)

get_all_models()

get_latest_model_record(model_name)

load_model_from_record(record)

11. Training API
Routes are defined in app/routers/training.py.

11.1 Train Single Model – POST /training/train
Request type: multipart/form-data

Fields:

Field	Type	Description
file	file (CSV)	e.g. private_lessons_data.csv
model_name	string	"linear", "decision_tree", "random_forest"
model_params	string (JSON)	Optional hyperparameters JSON

Requirements:

Valid JWT

Enough tokens (1 token)

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
11.2 Train Multiple Models – POST /training/train_multi
Trains multiple model types on the same dataset

model_names is a JSON list, e.g. ["linear", "random_forest"]

Returns metrics for each trained model

12. Prediction API
Routes are defined in app/routers/prediction.py.

12.1 List Models – GET /models/
Returns all trained models with their metrics (R², MAE, MSE, RMSE).

Example:

json
Copy code
{
  "models": [
    {
      "model_id": 1,
      "model_name": "linear",
      "model_type": "LinearRegression",
      "trained_at": "2025-12-07T10:18:12.390152",
      "r2": 0.96,
      "mae": 5.17,
      "mse": 52.83,
      "rmse": 7.27
    }
  ]
}
12.2 Predict – POST /models/predict/{model_name}
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
The prediction value is rounded to two decimal places.

13. Streamlit Admin Dashboard
File: tokens_dashboard.py

The dashboard:

Connects to the same SQLite database

Loads the users table via pandas

Displays:

All users

Tokens per user

Total tokens in the system

Run:

bash
Copy code
python -m streamlit run tokens_dashboard.py
14. Testing the API with Swagger
Start FastAPI server:

bash
Copy code
uvicorn app.main:app --reload
Open Swagger UI in the browser:

text
Copy code
http://127.0.0.1:8000/docs
Recommended manual test flow:

POST /auth/signup – create new user

POST /auth/login – get access_token

Click Authorize and paste the token (without Bearer)

POST /auth/add_tokens – add tokens (e.g. 20)

POST /training/train – upload private_lessons_data.csv and train a model

GET /models/ – check trained models and metrics

POST /models/predict/linear – send JSON data and get a price prediction

15. Future Improvements
Add more ML models (XGBoost, SVR, Gradient Boosting)

Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

Improve error messages and validation

Store full training history per user

Extend Streamlit dashboard with charts

Add automated tests with pytest

Add Docker configuration for deployment

markdown
Copy code
