🌟 Machine Learning & FastAPI — Final Project

מערכת מלאה ל־Machine Learning המשלבת FastAPI, אימות JWT, ניהול טוקנים, אימון מודלים, חיזוי מחירים למורים פרטיים ולוח ניהול (Streamlit).

1. TL;DR (Summary)

שימוש בדataset קבוע: data/private_lessons_data.csv

אימון מודלים: Linear Regression, Decision Tree, Random Forest

קריאות API מאובטחות עם JWT

כל פעולה צורכת טוקנים

שמירת מודלים כ־.pkl + מטא־דאטה JSON

Dashboard של Streamlit להצגת משתמשים וטוקנים

2. System Flow

משתמש נרשם → /auth/signup

מתחבר → מקבל JWT → /auth/login

מעלה dataset ומאמן מודל → /training/train

המודל נשמר + נרשמת היסטוריה

משתמש שולח בקשת חיזוי → /models/predict/{model_name}

המערכת טוענת את המודל האחרון ומחזירה מחיר

כל פעולה צורכת טוקנים לפי הגדרה

מנהל המערכת רואה משתמשים וטוקנים ב־Streamlit

3. Project Structure
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
│   ├── (saved .pkl models)
│   └── models_metadata.json
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md

4. Dataset
4.1 Dataset Structure (private_lessons_data.csv)
Column	Type
subject	string
student_level	string
lesson_minutes	int
teacher_experience_years	int
is_online	string
city	string
teacher_age	int
lesson_price (label)	float
Feature Columns Used for Training
["subject", "student_level", "lesson_minutes",
 "teacher_experience_years", "is_online", "city"]

Target Column
"lesson_price"

5. Jupyter Notebook (project_info.ipynb)

ה-notebook מספק:

טעינת הנתונים

df.head(), df.info(), df.describe()

גרפים באמצעות seaborn/matplotlib:

התפלגות מחירים

התפלגות משך שיעור

ניסיון מורים

שונות לפי עיר/נושא/רמת תלמיד

Heatmap קורלציות

דוגמה לאימון מודל רגרסיה

מדדים: R², MAE, RMSE

6. Technologies
Component	Technology
API	FastAPI
Authentication	JWT (python-jose)
Password Hashing	bcrypt
ML Engine	scikit-learn
Data Processing	pandas
Model Storage	joblib + JSON
Database	SQLite
Dashboard	Streamlit
Python Version	3.x
7. Installation
7.1 Clone the Repository
git clone https://github.com/karinshaham1302/19.10.2025.git
cd 19.10.2025

7.2 Create Virtual Environment

Windows

python -m venv .venv
.venv\Scripts\activate


macOS / Linux

python -m venv .venv
source .venv/bin/activate

7.3 Install Required Packages
pip install -r requirements.txt

8. Running the FastAPI Server
uvicorn app.main:app --reload


Endpoints:

API root → http://127.0.0.1:8000/

Swagger UI → http://127.0.0.1:8000/docs

9. Authentication & Token System
9.1 Token Costs (config.py)
Action	Tokens
Train model	1
Train multiple models	1
Predict	5
9.2 Auth Flow

Sign up → /auth/signup

Login → /auth/login

קבלת JWT

ב־Swagger → "Authorize" → הדבקת רק ה־token

כל בקשה מוגנת עובדת

9.3 Available Endpoints
Method	Endpoint	Description
POST	/auth/signup	Create user
POST	/auth/login	Get JWT token
GET	/auth/tokens	Check token balance
POST	/auth/add_tokens	Add tokens
DELETE	/auth/remove_user	Delete user
10. Model Service (Machine Learning Logic)

נמצא: app/model_service.py

אחריות עיקרית:

בדיקת תקינות dataset

בניית preprocessing (OneHotEncoder + numeric passthrough)

יצירת מודלים:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

אימון + שמירת המדדים:

R²

MAE

MSE

RMSE

שמירת מודל כ־.pkl וכתיבת metadata לקובץ JSON

פונקציות עיקריות:

train_model()

get_all_models()

get_latest_model_record()

load_model_from_record()

11. Training API
11.1 Train a Single Model

POST /training/train

Form-data fields:

Field	Type
file	CSV file
model_name	string
model_params	JSON (optional)

דוגמה לתשובה:

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

11.2 Train Multiple Models

POST /training/train_multi

מאמן מספר מודלים על אותו dataset

מחזיר טבלה עם מדדי כל מודל

12. Prediction API
12.1 List All Models

GET /models/

12.2 Predict

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

13. Streamlit Dashboard

הרצת הדשבורד:

python -m streamlit run tokens_dashboard.py


מציג:

כל המשתמשים

כמות הטוקנים

סטטיסטיקות מערכת

14. Future Improvements

תמיכה במודלים מתקדמים (XGBoost, SVR, Gradient Boosting)

Hyperparameter Tuning

Error handling חכם יותר

היסטוריית מודלים לפי משתמש

Dashboard עם גרפים

בדיקות אוטומטיות (pytest)

Docker ל־deployment
