# Machine Learning & FastAPI Final Project

This project implements a complete machine-learning pipeline using FastAPI, scikit-learn, JWT authentication, a token-based usage system, and a Streamlit admin dashboard.

---

## 1. TL;DR (Short Summary)

- Fixed dataset: `data/private_lessons_data.csv`  
- Train ML models: **Linear Regression**, **Decision Tree**, **Random Forest**  
- Make predictions through authenticated API calls  
- JWT authentication with token balance per user  
- Models saved as `.pkl` files with full metadata  
- Streamlit dashboard displays users + token balance  

---

## 2. How It Works (System Flow)

1. User signs up → `POST /auth/signup`  
2. User logs in → receives JWT token (`POST /auth/login`)  
3. User uploads dataset → train model (`POST /training/train`)  
4. Model is trained, evaluated, saved under `models/`  
5. User requests prediction → `POST /models/predict/{model_name}`  
6. System loads latest model, returns predicted price  
7. Each action consumes tokens  
8. Admin can view dashboard in Streamlit  

---

## 3. Project Structure

19.10.2025/
│
├── app/
│ ├── init.py
│ ├── main.py
│ ├── config.py
│ ├── database.py
│ ├── schemas.py
│ ├── auth_service.py
│ ├── model_service.py
│ └── routers/
│ ├── auth.py
│ ├── training.py
│ └── prediction.py
│
├── data/
│ └── private_lessons_data.csv
│
├── models/
│ └── saved .pkl models + metadata JSON
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md

csharp
Copy code

---

## 4. Dataset and Notebook

### 4.1 Dataset: `private_lessons_data.csv`

| Column                     | Type   |
|----------------------------|--------|
| subject                    | string |
| student_level              | string |
| lesson_minutes             | int    |
| teacher_experience_years   | int    |
| is_online                  | string |
| city                       | string |
| teacher_age                | int    |
| lesson_price *(label)*     | float  |

**Feature columns**

```python
["subject", "student_level", "lesson_minutes",
 "teacher_experience_years", "is_online", "city"]
Label column

python
Copy code
"lesson_price"
4.2 Notebook: project_info.ipynb
Includes:

Dataset loading

EDA: distributions, correlations, summary

Linear Regression example

Metrics: R², MAE, RMSE (rounded to 2 decimals)

Visualizations using seaborn/matplotlib

5. Technologies
Component	Technology
API	FastAPI
Auth	JWT (python-jose)
Hashing	bcrypt
ML	scikit-learn
Data	pandas
Storage	joblib + SQLite
Dashboard	Streamlit
Python	3.x

6. Installation
6.1 Clone
bash
Copy code
git clone https://github.com/karinshaham1302/19.10.2025.git
cd 19.10.2025
6.2 Virtual Environment
Windows

bash
Copy code
python -m venv .venv
.venv\Scripts\activate
macOS/Linux

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

API → http://127.0.0.1:8000/

Swagger UI → http://127.0.0.1:8000/docs

8. Authentication & Tokens
8.1 Token Costs
Action	Tokens
Train model	1
Train multiple	1
Predict	5

8.2 Flow
POST /auth/signup

POST /auth/login — copy the access_token

In Swagger → Authorize → paste only the token (no “Bearer”)

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

Metrics: R², MAE, MSE, RMSE (rounded 2 decimals)

Saving model + metadata JSON

Utility functions:

get_all_models()

get_latest_model_record()

load_model_from_record()

10. Training API
10.1 Train Single Model
POST /training/train

Fields:

file — CSV file

model_name — "linear", "decision_tree", "random_forest"

model_params — optional JSON

Response example:

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

11.2 Predict
POST /models/predict/{model_name}

Request:

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

Users

Token balances

System totals

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
Add more ML models (XGBoost, SVR, Gradient Boosting)

Hyperparameter tuning

Better error messages

Model history per user

Streamlit charts

Add pytest tests

Docker containerization
