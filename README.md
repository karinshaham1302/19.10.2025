This project implements an end-to-end machine learning system for predicting private lesson prices using FastAPI, scikit-learn, JWT authentication, and a token-based usage system.
A Streamlit admin dashboard is also included.

1. TL;DR (Short Summary)

Fixed dataset: data/private_lessons_data.csv

Train ML models: Linear Regression, Decision Tree, Random Forest

Make predictions through authenticated API calls

JWT-based authentication with token balance per user

Models are saved as .pkl files with full metadata and metrics

Streamlit dashboard displays users and their token balance

2. How It Works (System Flow)

User signs up → POST /auth/signup

User logs in → POST /auth/login → receives a JWT access token

User uploads private_lessons_data.csv → POST /training/train

The model is trained, evaluated, saved under models/ (+ JSON metadata)

User sends a prediction request → POST /models/predict/{model_name}

The system loads the latest model and returns the predicted price

Each action consumes tokens according to configured costs

Admin can view all users and tokens via the Streamlit dashboard

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

This is a fixed, realistic dataset of private lessons, used both:

in the FastAPI training endpoint (/training/train), and

in the Jupyter notebook (project_info.ipynb) for EDA and example model training.

Main columns:

subject

student_level

lesson_minutes

teacher_experience_years

is_online

city

teacher_age

lesson_price (label)

Feature columns used by the ML pipeline (FEATURE_COLUMNS):

["subject", "student_level", "lesson_minutes",
 "teacher_experience_years", "is_online", "city"]


Target column (TARGET_COLUMN):

"lesson_price"

4.2 Notebook: project_info.ipynb

The notebook includes:

Loading the dataset with pandas

df.head(), df.info(), df.describe(include="all")

Visualizations (matplotlib / seaborn):

Distribution of lesson prices

Distribution of lesson durations

Teacher experience distribution

Price by subject, student level, city, and online/offline

Correlation heatmap for numeric features

A simple reference model training (Linear Regression):

Train/test split

Model training

Metrics: R², MAE, RMSE (printed with 2 decimal places)

A Summary of Findings section describing the main insights.

This notebook serves as the EDA and human-readable explanation of the same data used by the API.

5. Technologies
Component	Technology
Web API	FastAPI
Auth	JWT (python-jose)
Passwords	bcrypt (passlib)
ML	scikit-learn
Data handling	pandas
Model storage	joblib
Database	SQLite
Admin UI	Streamlit
Python version	3.x

All dependencies are defined in requirements.txt.

6. Installation
6.1 Clone the repository
git clone https://github.com/karinshaham1302/19.10.2025.git
cd 19.10.2025

6.2 Create and activate virtual environment

On Windows (PowerShell):

python -m venv .venv
.venv\Scripts\activate


On macOS / Linux:

python -m venv .venv
source .venv/bin/activate

6.3 Install dependencies
pip install -r requirements.txt

7. Running the FastAPI Server

From the project root (19.10.2025/), with the virtual environment activated:

uvicorn app.main:app --reload


The server will be available at:

API root: http://127.0.0.1:8000/

Swagger UI: http://127.0.0.1:8000/docs

8. Authentication & Tokens

Users and tokens are stored in an SQLite database: data/app.db.

8.1 Token Costs (from config.py)
Action	Tokens
Train single model	1
Train multiple	1
Prediction	5
8.2 Auth Flow

Sign up – POST /auth/signup

Log in – POST /auth/login → returns access_token

In Swagger UI (/docs), click Authorize and paste only the token (without the word Bearer).

All protected endpoints will now use that JWT automatically.

8.3 Main Auth Endpoints
Method	Endpoint	Description
POST	/auth/signup	Register new user
POST	/auth/login	Login, returns JWT access token
GET	/auth/tokens	Get current token balance (logged-in)
POST	/auth/add_tokens	Add tokens to logged-in user
DELETE	/auth/remove_user	Delete user by username + password
9. Model Service (ML Logic)

The core machine learning logic is implemented in app/model_service.py.

Responsibilities:

Validate dataset (validate_dataset):
Checks that all required columns exist.

Preprocessing (build_preprocessor):
Uses ColumnTransformer and OneHotEncoder for categorical features, and passthrough for numeric features.

Model creation (create_model):
Supported models:

"linear" → LinearRegression

"decision_tree" → DecisionTreeRegressor

"random_forest" → RandomForestRegressor

Training (train_model):

Train/test split (test_size=0.2, random_state=42)

Fit pipeline

Compute metrics:

R²

MAE

MSE

RMSE (square root of MSE)

All metrics are rounded to 2 decimal places and stored as floats.

Saving (save_model_with_metadata):

Saves pipeline as .pkl under models/

Writes metadata (model id, name, type, features, metrics, path, etc.) into models_metadata.json.

Utility functions:

get_all_models() – returns list of all trained models from metadata

get_latest_model_record(model_name) – returns the latest model with that name

load_model_from_record(record) – loads a stored pipeline from disk

10. Training API

Training endpoints are located in app/routers/training.py.

10.1 Train Single Model – POST /training/train

Request type: multipart/form-data

Fields:

file – CSV file (e.g. private_lessons_data.csv)

model_name – "linear", "decision_tree", or "random_forest"

model_params – optional JSON string with hyperparameters (e.g. {"max_depth": 5})

Requirements:

Valid JWT (logged-in user)

Enough tokens (1 token, TRAIN_TOKENS_COST)

Example of response (simplified):

{
  "status": "success",
  "message": "Model was trained successfully and is ready for predictions.",
  "model_info": {
    "model_id": 1,
    "model_name": "linear",
    "model_type": "LinearRegression",
    "trained_at": "2025-12-07T10:18:12.390152",
    "label_column": "lesson_price",
    "n_samples": 100,
    "metrics": {
      "r2": 0.96,
      "mae": 5.17,
      "mse": 52.83,
      "rmse": 7.27
    },
    "model_path": "app/models/linear_20251207_101812.pkl"
  }
}

10.2 Train Multiple Models – POST /training/train_multi

Trains multiple model types on the same dataset.

model_names is a JSON list of strings (e.g. ["linear", "random_forest"]).

Returns:

A list of trained models with their metrics

The best model name according to R²

Consumes 1 token (TRAIN_MULTI_TOKENS_COST).

11. Prediction API

Prediction endpoints are defined in app/routers/prediction.py.

11.1 List All Trained Models – GET /models/

Requires a valid JWT.

Returns a list of all trained models and their metrics:

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

11.2 Predict with Latest Model – POST /models/predict/{model_name}

Example path: /models/predict/linear

Request body:

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


The endpoint:

Loads the latest model with the given model_name

Converts the data dict into a one-row DataFrame

Applies the full pipeline (preprocessing + model)

Returns the predicted price

Deducts 5 tokens (PREDICT_TOKENS_COST)

Example response:

{
  "model_name": "linear",
  "model_id": 2,
  "prediction": 163.04
}

12. Streamlit Admin Dashboard

File: tokens_dashboard.py

Purpose:

Connects to the same SQLite database (data/app.db)

Loads the users table into a pandas DataFrame

Displays:

All users

Their token balances

Aggregate metrics (total users, total tokens)

Run:

python -m streamlit run tokens_dashboard.py


This opens a local web page with the admin dashboard.

13. Testing the API with Swagger

Start the FastAPI server:

uvicorn app.main:app --reload


Open Swagger UI:

http://127.0.0.1:8000/docs

Recommended flow:

POST /auth/signup – create a new user

POST /auth/login – get access_token

Click Authorize → paste the token (without Bearer)

POST /auth/add_tokens – add tokens (for example, 20)

POST /training/train – upload private_lessons_data.csv and train a model

GET /models/ – check that the model appears with metrics

POST /models/predict/linear – send JSON data and get a predicted price

14. Future Improvements

Possible extensions:

Add more model types (e.g. XGBoost, SVR, Gradient Boosting)

Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

Add model comparison dashboard

Improve error handling and messages

Track training history per user

Extend Streamlit dashboard with charts and usage analytics

Add pytest-based tests for:

Routers

Auth logic

Model training/prediction

Add Dockerfile for easier deployment
