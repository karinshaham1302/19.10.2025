Machine Learning & FastAPI Final Project
Quick Summary (TL;DR)

This project implements a complete machine-learning pipeline delivered over a FastAPI server.
It supports:

Uploading a CSV dataset

Training ML models (Linear Regression, Decision Tree, Random Forest)

Saving models with metadata

Making predictions using the latest trained model

Full authentication using JWT

Token-based usage control (1 token for training, 5 tokens for prediction)

A Streamlit dashboard for monitoring users and token balances

A predefined dataset structure for private-lessons pricing

Start the API server:

uvicorn app.main:app --reload


Open the API interface:

http://127.0.0.1:8000/docs


Start the Streamlit dashboard:

python -m streamlit run tokens_dashboard.py

How the System Works
API Workflow Overview

The backend is composed of three main workflows: authentication, model training, and prediction.

1. Authentication Workflow

User registers via /auth/signup.

User logs in via /auth/login.

The server returns a JWT token.

The user pastes only the token into the Swagger "Authorize" window.

All protected endpoints require this token.

Purpose:

Secure model operations

Identify each user

Ensure that training/prediction actions are tracked

2. Token System Workflow

Each operation costs tokens:

Operation	Cost
Model Training	1 token
Prediction	5 tokens

Process:

Before any action, the system checks whether the user has enough tokens.

If not, the API returns a 403 error.

After a successful operation, the tokens are deducted.

3. Model Training Workflow (POST /training/train)

User uploads the CSV file.

Backend validates the dataset schema.

Preprocessing step:

OneHotEncoder for categorical columns

Passthrough for numeric columns

The selected model is created (Linear Regression, Decision Tree, Random Forest).

The model is trained on the dataset.

Evaluation metrics are computed:

R²

MAE

MSE

RMSE
(all rounded to two decimal places)

The trained model is saved as a .pkl file.

Metadata is stored in models_metadata.json.

Output includes model ID, full metadata, and evaluation metrics.

4. Prediction Workflow (POST /models/predict/{model_name})

User submits a JSON object with feature values.

System verifies JWT authentication.

System verifies user tokens (requires 5).

The backend loads the latest model for the given model name.

A prediction is generated, rounded to two decimal places.

System deducts 5 tokens.

Returns the predicted value.

5. Streamlit Monitoring Dashboard

A small dashboard used for viewing system usage and user token balances.

Run with:

python -m streamlit run tokens_dashboard.py


Displays:

All users

Token counts per user

Dataset and Notebook
CSV Dataset

The project uses a fixed dataset:

data/private_lessons_data.csv


It contains simulated private-lesson pricing data with the following columns:

subject

student_level

lesson_minutes

teacher_experience_years

is_online

city

teacher_age

lesson_price

This dataset is used for:

Exploratory Data Analysis (EDA)

Model training

Model evaluation

Jupyter Notebook (EDA)

Full exploratory analysis appears in:

project_info.ipynb


It includes:

Data loading

Summary statistics

Visualizations (price distribution, correlations, subject effects, and more)

A small ML model evaluation replicating the API logic

Project Structure
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
│   ├── routers/
│   │   ├── auth.py
│   │   ├── training.py
│   │   └── prediction.py
│
├── data/
│   └── private_lessons_data.csv
│
├── models/
│   └── (automatically saved .pkl model files)
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md

Installation & Usage
1. Create and Activate Virtual Environment
python -m venv .venv
.venv\Scripts\activate        (Windows)


Install dependencies:

pip install -r requirements.txt

2. Run the API Server
uvicorn app.main:app --reload


Open Swagger UI:

http://127.0.0.1:8000/docs

3. Authentication Steps

Sign up: /auth/signup

Log in: /auth/login

Copy the returned token

Click “Authorize” in Swagger and paste the token only

4. Train a Model

Upload the CSV and choose a model type via /training/train.

5. Get Available Models
GET /models

6. Make a Prediction

Send a JSON with feature values to /models/predict/{model_name}.

Future Improvements

Possible enhancements:

Support additional model types (XGBoost, Neural Networks).

Add data preprocessing options (scaling, feature selection).

Add detailed training history visualization.

Add admin panel for user management.

Extend Streamlit dashboard to include:

Logs viewer

Model comparison

Error analysis

Automatic dataset validation and anomaly detection.

CI/CD pipeline for testing and deployment.

Docker support for containerized deployment.
