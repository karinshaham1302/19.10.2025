# Machine Learning & FastAPI Final Project

## Quick Summary (TL;DR)

This project implements a complete machine learning pipeline exposed via a FastAPI server.  
It includes:

- A predefined CSV dataset for private-lesson pricing
- A full EDA notebook in Jupyter
- Model training (Linear Regression, Decision Tree, Random Forest)
- Saving models with metadata
- Predictions using the latest trained model
- JWT-based authentication
- Token-based usage control (1 token for training, 5 for prediction)
- A Streamlit dashboard for monitoring users and token balances

---

## How the System Works

The system is built around three main workflows: **authentication**, **model training**, and **prediction**, plus an **admin dashboard**.

### 1. Authentication

Endpoints:

- `POST /auth/signup` – register a new user
- `POST /auth/login` – log in and receive a JWT access token
- `GET /auth/tokens` – check current token balance
- `POST /auth/add_tokens` – add tokens to the current user
- `DELETE /auth/remove_user` – delete a user (with password confirmation)

Flow:

1. User signs up with username and password.
2. User logs in and gets a JWT token.
3. The user pastes **only the token** into the Swagger **Authorize** dialog.
4. All protected endpoints use this token to identify the user and control access.

### 2. Token System

Each operation consumes tokens:

| Operation            | Cost (tokens) |
|----------------------|---------------|
| Model training       | 1             |
| Training multiple models (train_multi) | 1 |
| Prediction           | 5             |

Before performing an action, the API checks:

- That the user is authenticated (valid JWT)
- That the user has enough tokens

If there are not enough tokens → HTTP 403 error.  
After a successful action, tokens are deducted and the new balance is stored in the database.

### 3. Model Training

Main endpoint:

- `POST /training/train`

Input (multipart/form-data):

- `file`: CSV file (`private_lessons_data.csv`)
- `model_name`: one of:
  - `linear`
  - `decision_tree`
  - `random_forest`
- `model_params` (optional, JSON string): hyperparameters for the chosen model

Internal steps:

1. Load the CSV into a pandas DataFrame.
2. Validate that the dataset contains all required columns.
3. Split into train/test sets.
4. Build a preprocessing pipeline:
   - `OneHotEncoder` for categorical columns
   - `passthrough` for numerical columns
5. Create a model pipeline with the chosen estimator:
   - `LinearRegression`
   - `DecisionTreeRegressor`
   - `RandomForestRegressor`
6. Train the model.
7. Evaluate on the test set with:
   - R² (coefficient of determination)
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   (all metrics are rounded to two decimal places)
8. Save the trained pipeline as a `.pkl` file under `models/`.
9. Store model metadata in `app/models/models_metadata.json`:
   - ID, name, type
   - Training timestamp
   - Features and label
   - Metrics
   - Path to the saved model file

The `POST /training/train` response includes:

- `status` and `message`
- `model_info` – full metadata, including metrics and the `model_id`

### 4. Prediction

Main endpoint:

- `POST /models/predict/{model_name}`

Usage:

1. Ensure the user has at least 5 tokens.
2. Call `POST /models/predict/linear` (or other `model_name`) with a JSON body:

   ```json
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
The backend:

Loads the latest trained model with that model_name.

Builds a one-row DataFrame from data.

Runs .predict(...).

Rounds the prediction to two decimal places.

Deducts 5 tokens from the current user.

Response example:

json
Copy code
{
  "model_name": "linear",
  "model_id": 2,
  "prediction": 163.04
}
There is also:

GET /models – list all trained models and their metrics.

5. Streamlit Dashboard
File: tokens_dashboard.py

The dashboard connects directly to the same SQLite database used by the API.

It displays:

A table of all users

Token balance per user

Simple summary metrics (total users, total tokens in the system)

Run it with:

bash
Copy code
python -m streamlit run tokens_dashboard.py
Open the URL shown in the terminal (usually http://localhost:8501).

Dataset and Jupyter Notebook
CSV Dataset
Path:

text
Copy code
data/private_lessons_data.csv
This is a fixed dataset simulating private lesson pricing, with columns:

subject

student_level

lesson_minutes

teacher_experience_years

is_online

city

teacher_age

lesson_price

Uses:

Training the machine learning models in the API

EDA and visualizations in the Jupyter notebook

Jupyter Notebook (EDA and Model Check)
Path:

text
Copy code
project_info.ipynb
Contents:

Loading the CSV dataset

df.describe(include="all") and basic statistics

Distribution of lesson prices

Teacher experience and lesson duration distributions

Correlation heatmap for numeric features

Box plots and bar charts by subject, level, and city

A small linear regression model trained inside the notebook, with:

Train/test split

Evaluation metrics

Interpretation of the results

The notebook is the “report” part of the project, demonstrating that the dataset is consistent and that the learned model fits the data.

Project Structure
text
Copy code
19.10.2025/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Paths, JWT config, token pricing
│   ├── database.py          # SQLite connection and user table
│   ├── schemas.py           # Pydantic request/response models
│   ├── auth_service.py      # Authentication, JWT, token checks
│   ├── model_service.py     # ML training, saving models, metadata
│   └── routers/
│       ├── auth.py          # /auth/... endpoints
│       ├── training.py      # /training/... endpoints
│       └── prediction.py    # /models/... endpoints
│
├── data/
│   └── private_lessons_data.csv   # Main dataset used by both API and notebook
│
├── models/
│   └── (generated .pkl files after training)
│
├── project_info.ipynb       # EDA + mini ML experiment in Jupyter
├── tokens_dashboard.py      # Streamlit dashboard for users & tokens
├── requirements.txt         # Python dependencies
└── README.md
Installation and Usage
1. Create and Activate Virtual Environment
bash
Copy code
python -m venv .venv
On Windows:

bash
Copy code
.venv\Scripts\activate
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. Run the FastAPI Server
bash
Copy code
uvicorn app.main:app --reload
Open the API docs (Swagger UI):

text
Copy code
http://127.0.0.1:8000/docs
4. Authentication Flow
Go to POST /auth/signup

Register a user, e.g.:

json
Copy code
{
  "username": "student_demo",
  "password": "ab1234"
}
Go to POST /auth/login

Log in with the same credentials.

Copy the access_token value from the response.

Click the Authorize button in Swagger (top right).

Paste the token only (no “Bearer ” prefix).

Confirm.

Now all protected endpoints will use this token.

5. Manage Tokens
Check current tokens:
GET /auth/tokens

Add tokens (for testing):
POST /auth/add_tokens

json
Copy code
{
  "amount": 20
}
6. Train a Model
Go to:

POST /training/train

Fill in:

Upload data/private_lessons_data.csv as file

Set model_name to one of:

linear

decision_tree

random_forest

Optional: model_params as JSON string (or leave empty)

Execute and check the response for:

model_id

metrics (r2, mae, mse, rmse – all with two decimal places)

7. List Trained Models
Go to:

GET /models

You will get a list of all models and their metrics.

8. Make a Prediction
Go to:

POST /models/predict/{model_name} (e.g. /models/predict/linear)

Request body example:

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
9. Run the Streamlit Dashboard
In a separate terminal (with the virtual environment active):

bash
Copy code
python -m streamlit run tokens_dashboard.py
Open the URL shown in the console (e.g. http://localhost:8501)
You should see:

A table of users

Their token balances

Basic summary metrics

Future Improvements
Some possible extensions for this project:

Support additional model types (e.g. XGBoost, neural networks)

Add more flexible configuration for:

Feature selection

Train/test split

Hyperparameter tuning

Extend dataset validation and add automatic checks for:

Missing values

Outliers

Class imbalance (for classification tasks)

Add an admin-only FastAPI router for:

Viewing logs

Resetting user tokens

Deleting models

Extend the Streamlit dashboard to:

Compare models by metrics

Plot training history and errors

Add automated tests and CI/CD pipeline

Add Docker support for easier deployment

markdown
Copy code

---

## איך להכניס את זה ל-GitHub (רק דרך הדפדפן)

1. היכנסי ל-GitHub לריפו שלך:  
   `https://github.com/karinshaham1302/19.10.2025`

2. ברשימת הקבצים, לחצי על `README.md`.

3. בצד ימין למעלה של הקובץ, לחצי על כפתור העיפרון **Edit this file**.

4. מחקי את כל הטקסט הקיים (Ctrl+A → Delete).

5. חזרי לכאן, סמני את כל ה-README שבקודבלוק הגדול למעלה (מ`# Machine Learning & FastAPI Final Project` ועד סוף הקובץ), והעתיקי.

6. הדביקי ב-GitHub בתוך חלון העריכה.

7. מעל הטופס של ה-Commit, לחצי על לשונית **Preview** כדי לראות איך זה נראה מעוצב:
   - כותרות גדולות
   - קוד בצבע אפור
   - עץ תיקיות מיושר  
   אם זה נראה טוב – ממשיכים.

8. למטה, בשדה **Commit message** אפשר לכתוב משהו כמו:  
   `Update README with full project documentation`

9. סמני את האפשרות **Commit directly to the main branch**.

10. לחצי על **Commit changes**.

וזהו – ה-README החדש יופיע יפה, מסודר, עם ה-CSV וה-Jupyter, בלי סמלים מעצבנים ובלי בלגן. 💪  

אם אחרי שתדביקי אותו משהו עדיין ייראה לא טוב (שורות שנשברות מוזר, קטע מסוים לא ברור), תצלמי מסך ונעבור על זה ביחד שורה-שורה.

