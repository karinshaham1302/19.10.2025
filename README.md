# Machine Learning & FastAPI Final Project

This project implements an end-to-end machine learning system for predicting private lesson prices, exposed via a FastAPI backend.  
The system supports:

- Training regression models on a predefined private-lessons dataset
- Making predictions via authenticated REST API calls
- Managing users and token-based usage
- Viewing users and tokens via a Streamlit admin dashboard

---

## 1. TL;DR (Short Summary)

- Fixed dataset: private lessons with predefined schema (`private_lessons_data.csv`)
- Models: Linear Regression, Decision Tree, Random Forest (via scikit-learn)
- Training and prediction are exposed as FastAPI endpoints
- JWT authentication for all protected endpoints
- Each action consumes tokens (train/predict)
- Trained models are saved to disk (`app/models/`) with JSON metadata
- Exploratory Data Analysis (EDA) and a simple training example are documented in `project_info.ipynb`
- Admin dashboard (Streamlit) shows all users and their token balances

---

## 2. How It Works (High-Level Flow)

1. User signs up (`/auth/signup`)
2. User logs in (`/auth/login`) and receives a JWT access token
3. Authenticated user uploads `private_lessons_data.csv` to train a model (`/training/train`)
4. The server:
   - Validates the dataset schema
   - Builds preprocessing (One-Hot encoding + numeric passthrough)
   - Trains the selected model
   - Evaluates it (R², MAE, MSE, RMSE – rounded to 2 decimal places)
   - Saves the trained pipeline to `app/models/` as a `.pkl` file
   - Stores metadata in `models_metadata.json`
5. User requests predictions via `/models/predict/{model_name}`
6. The API loads the latest trained model of that type and returns a predicted lesson price
7. Every training/prediction call consumes tokens from the user’s account
8. The Streamlit dashboard reads the same SQLite DB and displays all users and tokens

---

## 3. Project Structure

```text
19.10.2025/
│
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI entry point
│   ├── config.py             # Paths, JWT config, token pricing
│   ├── database.py           # SQLite connection and user table
│   ├── schemas.py            # Pydantic models (requests / responses)
│   ├── auth_service.py       # JWT, password hashing, token checks
│   ├── model_service.py      # ML pipeline: training, saving, loading models
│   └── routers/
│       ├── auth.py           # /auth/... endpoints (signup, login, tokens, etc.)
│       ├── training.py       # /training/... endpoints (train, train_multi)
│       └── prediction.py     # /models/... endpoints (list_models, predict)
│
├── data/
│   └── private_lessons_data.csv   # Main dataset used for training and EDA
│
├── models/
│   └── (saved .pkl model files + models_metadata.json)
│
├── project_info.ipynb        # Jupyter notebook: EDA + simple ML training example
├── tokens_dashboard.py       # Streamlit admin dashboard for users/tokens
├── requirements.txt          # Python dependencies
└── README.md                 # This file

## 4. Dataset and Jupyter Notebook

## 4.1 Dataset: private_lessons_data.csv
The project uses a fixed, realistic dataset of private lessons, stored at:

text
Copy code
data/private_lessons_data.csv
Main columns:

subject

student_level

lesson_minutes

teacher_experience_years

is_online

city

teacher_age

lesson_price (label)

For the machine learning model, the following are used:

Features (FEATURE_COLUMNS in model_service.py):
["subject", "student_level", "lesson_minutes", "teacher_experience_years", "is_online", "city"]

Target (TARGET_COLUMN):
"lesson_price"

The CSV file is also used both in:

The FastAPI training endpoint (/training/train)

The Jupyter notebook (project_info.ipynb) for EDA and a reference training example

## 4.2 Notebook: project_info.ipynb
The notebook includes:

Loading the dataset with pandas

df.head() + df.info() + df.describe(include="all")

Visualizations (using matplotlib/seaborn):

Distribution of lesson prices

Distribution of lesson durations

Teacher experience distribution

Price by subject, student level, city, and online/offline

Correlation heatmap for numeric features

A simple reference model training:

Train/test split

Linear Regression

Evaluation: R², MAE, RMSE (printed with 2 decimal places)

A summary section describing the main insights from the EDA

The notebook serves as a data exploration and “human-readable” explanation of the same data used by the API.


## 5. Technologies
Component	Technology
Language	Python 3.x
Web framework	FastAPI
Data handling	pandas
ML	scikit-learn
Model storage	joblib
Auth	JWT (python-jose) + bcrypt
DB	SQLite
Admin UI	Streamlit

Dependencies are listed in requirements.txt.


## 6. Environment Setup and Installation

## 6.1 Clone the Repository
bash
Copy code
git clone https://github.com/karinshaham1302/19.10.2025.git
cd 19.10.2025
## 6.2 Create and Activate Virtual Environment
On Windows (PowerShell):

bash
Copy code
python -m venv .venv
.venv\Scripts\activate
On macOS / Linux:

bash
Copy code
python -m venv .venv
source .venv/bin/activate

## 6.3 Install Dependencies
bash
Copy code
pip install -r requirements.txt

## 7. Running the FastAPI Server
In the project root (19.10.2025/), with the virtual environment activated:

bash
Copy code
uvicorn app.main:app --reload
The server will be available at:

API root: http://127.0.0.1:8000/

Swagger UI: http://127.0.0.1:8000/docs


## 8. Authentication and Token System
Users and tokens are stored in an SQLite database: data/app.db.

## 8.1 Token Pricing (from config.py)
Action	Constant	Cost (tokens)
Train single model	TRAIN_TOKENS_COST	1
Train multiple models	TRAIN_MULTI_TOKENS_COST	1
Prediction	PREDICT_TOKENS_COST	5

## 8.2 Auth Flow
Sign up: POST /auth/signup

Log in: POST /auth/login

Copy the returned access_token

In Swagger (/docs), click “Authorize” and paste only the token (without the word Bearer)

All protected endpoints will now use that JWT

## 8.3 Main Auth Endpoints
Method	Path	Description	Auth required
POST	/auth/signup	Register new user	No
POST	/auth/login	Login, returns JWT access token	No
GET	/auth/tokens	Get current token balance for logged-in user	Yes (JWT)
POST	/auth/add_tokens	Add tokens to logged-in user (simulated purchase)	Yes (JWT)
DELETE	/auth/remove_user	Delete user by username + password	No (request body auth)


## 9. Model Service (Machine Learning Logic)
The machine learning logic is implemented in app/model_service.py.

Main responsibilities:

Validating the dataset (validate_dataset)

Building preprocessing with ColumnTransformer and OneHotEncoder (build_preprocessor)

Creating models (create_model) – supported model names:

"linear" → LinearRegression

"decision_tree" → DecisionTreeRegressor

"random_forest" → RandomForestRegressor

Training (train_model):

Train/test split (default test_size=0.2, random_state=42)

Fitting the pipeline

Evaluating metrics:

R²

MAE

MSE

RMSE (square root of MSE)

All metrics are stored rounded to 2 decimal places

Saving model and metadata (save_model_with_metadata):

Saves .pkl file under app/models/

Stores metadata in models_metadata.json

Utility functions:

get_all_models() – returns full list of trained models (from metadata)

get_latest_model_record(model_name) – returns the latest model of a given name

load_model_from_record() – loads a saved pipeline from disk


## 10. Training API
Training endpoints are defined in app/routers/training.py.

## 10.1 Train Single Model
POST /training/train

Request type: multipart/form-data

Fields:

file – CSV file (e.g., private_lessons_data.csv)

model_name – "linear", "decision_tree", or "random_forest"

model_params (optional) – JSON string with hyperparameters (e.g., "{"max_depth": 5}")

Requires:

Valid JWT

Sufficient tokens (1 token)

Example response (simplified):

json
Copy code
{
  "status": "success",
  "message": "Model was trained successfully and is ready for predictions.",
  "model_info": {
    "model_id": 1,
    "model_name": "linear",
    "model_type": "LinearRegression",
    "trained_at": "2025-12-07T10:18:12.390152",
    "features_used": [
      "subject",
      "student_level",
      "lesson_minutes",
      "teacher_experience_years",
      "is_online",
      "city"
    ],
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
10.2 Train Multiple Models
POST /training/train_multi

Trains multiple model types on the same dataset

model_names is a JSON list of strings, for example: ["linear", "random_forest"]

Returns metrics for each model and indicates which had the best R²

Also consumes tokens (1 token per request according to TRAIN_MULTI_TOKENS_COST)


## 11. Prediction API
Prediction endpoints are defined in app/routers/prediction.py.

## 11.1 List All Trained Models
GET /models/

Requires JWT

Returns a list of all models and their metrics (R², MAE, MSE, RMSE)

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

## 11.2 Predict with Latest Model
POST /models/predict/{model_name}

Example path: /models/predict/linear

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
The endpoint:

Loads the latest saved model of that name

Converts the data dictionary into a one-row DataFrame

Applies the pipeline and returns the predicted price

Consumes 5 tokens (PREDICT_TOKENS_COST)

Example response:

json
Copy code
{
  "model_name": "linear",
  "model_id": 2,
  "prediction": 163.04
}
(rounded in the API to 2 decimal places).


## 12. Streamlit Admin Dashboard
File: tokens_dashboard.py

Purpose:

Connects directly to the same SQLite database (data/app.db)

Loads the users table into a pandas DataFrame

Displays:

Table of all users

Total number of users

Total number of tokens in the system

Run:

bash
Copy code
python -m streamlit run tokens_dashboard.py
This will open a local web page with the dashboard.


## 13. How to Test the API (Swagger)
Start FastAPI server:

bash
Copy code
uvicorn app.main:app --reload
Open Swagger UI:

text
Copy code
http://127.0.0.1:8000/docs
Test flow:

POST /auth/signup – create new user

POST /auth/login – get access_token

Click “Authorize” in Swagger and paste the token (without Bearer)

POST /auth/add_tokens – add some tokens (for example, 20)

POST /training/train – upload private_lessons_data.csv and train a model

GET /models/ – check that the model is listed with metrics

POST /models/predict/linear – send a JSON with data and get a price prediction


## 14. Future Improvements
Some possible extensions:

Add more model types (e.g., XGBoost, SVR, Gradient Boosting)

Add hyperparameter search (GridSearchCV / RandomizedSearchCV)

Add pagination and filters to list of models

Add better error handling and user-friendly error messages

Extend the Streamlit dashboard to:

Show training history per user

Display charts of token usage

Add tests (pytest) for:

Routers

Auth logic

Model training and prediction

Containerization with Docker for easier deployment

go
Copy code

