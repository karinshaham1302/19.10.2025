# Machine Learning & FastAPI — Final Project

This project implements an end-to-end machine learning system for predicting private-lesson prices using FastAPI, scikit-learn, JWT authentication, a token-based usage system, and a Streamlit admin dashboard.

---

## 1. TL;DR (Short Summary)

- Fixed dataset: `data/private_lessons_data.csv`
- Train ML models: Linear Regression, Decision Tree, Random Forest
- Make predictions via authenticated API calls
- JWT-based authentication with per-user token balance
- Models saved as `.pkl` + JSON metadata
- Streamlit dashboard for users & tokens

---

## 2. System Flow 

1. User signs up → `/auth/signup`
2. User logs in → receives JWT → `/auth/login`
3. User trains a model → `/training/train`
4. Model is saved under `models/`
5. User makes prediction → `/models/predict/{model}`
6. System loads latest model and returns price
7. Tokens deducted per action
8. Admin views everything in Streamlit dashboard

---

## 3. Project Structure

```text
19.10.2025/
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
│   ├── (.pkl model files)
│   └── models_metadata.json
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md
```

## 4. Dataset and Notebook

### 4.1 Dataset: `private_lessons_data.csv`

#### Schema

| Column | Type |
|--------|------|
| subject | string |
| student_level | string |
| lesson_minutes | int |
| teacher_experience_years | int |
| is_online | string |
| city | string |
| teacher_age | int |
| lesson_price | float |

#### Feature Columns
["subject", "student_level", "lesson_minutes",
"teacher_experience_years", "is_online", "city"]

yaml Copy code

---

### 4.2 Jupyter Notebook: `project_info.ipynb`

Includes:

- Dataset exploration (head/info/describe)
- EDA visualizations
- Correlations heatmap
- Linear Regression training
- Evaluation (R², MAE, RMSE)
- Insights summary

---

## 5. Technologies

| Component | Technology |
|-----------|------------|
| API | FastAPI |
| Auth | JWT |
| ML | scikit-learn |
| Data | pandas |
| Storage | joblib + JSON |
| DB | SQLite |
| Dashboard | Streamlit |
| Python Version | 3.x |

---

## 6. Installation

### 6.1 Clone the Repository
git clone https://github.com/karinshaham1302/19.10.2025.git
cd 19.10.2025

markdown
Copy code

### 6.2 Create Virtual Environment

**Windows**
python -m venv .venv
.venv\Scripts\activate

markdown
Copy code

**macOS/Linux**
python -m venv .venv
source .venv/bin/activate

shell
Copy code

### 6.3 Install Dependencies
pip install -r requirements.txt

yaml
Copy code

---

## 7. Running the FastAPI Server

### 7.1 Start Server
uvicorn app.main:app --reload

yaml
Copy code

### 7.2 Access API
- Root: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs

---

## 8. Authentication & Token System

### 8.1 Token Costs

| Action | Tokens |
|--------|--------|
| Train model | 1 |
| Train multiple | 1 |
| Predict | 5 |

### 8.2 Flow
1. Sign up → `/auth/signup`  
2. Log in → get JWT  
3. Authorize via Swagger  
4. Start training & predicting  

### 8.3 Auth Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /auth/signup | Create new user |
| POST | /auth/login | Get JWT token |
| GET | /auth/tokens | View token balance |
| POST | /auth/add_tokens | Add tokens |
| DELETE | /auth/remove_user | Delete user |

---

## 9. Model Service (ML Logic)

### 9.1 Responsibilities
- Validate dataset  
- Preprocessing pipeline  
- Train models  
- Produce metrics  
- Save model + metadata  
- Load model for predictions  

### 9.2 Preprocessing Pipeline
- Categorical → OneHotEncoder  
- Numeric → passthrough  

### 9.3 Supported Models

| Name | Algorithm |
|-------|-----------|
| linear | LinearRegression |
| decision_tree | DecisionTreeRegressor |
| random_forest | RandomForestRegressor |

### 9.4 Metrics (rounded to 2 decimals)

| Metric | Meaning |
|--------|---------|
| R² | accuracy |
| MAE | absolute error |
| MSE | squared error |
| RMSE | root mean squared |

Example:
{ "r2": 0.96, "mae": 5.17, "mse": 52.83, "rmse": 7.27 }

yaml
Copy code

---

## 10. Training API

### 10.1 Single Model — `/training/train`

Fields:

| Field | Type | Description |
|-------|-------|-------------|
| file | CSV | dataset |
| model_name | string | linear / decision_tree / random_forest |
| model_params | JSON | optional |

### 10.2 Multiple Models — `/training/train_multi`
Trains several models at once.  
Returns metrics for all models + best model.

---

## 11. Prediction API

### 11.1 List All Models — `/models/`
Returns metadata for every saved model.

### 11.2 Predict — `/models/predict/{model_name}`

Example body:
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

yaml
Copy code

---

## 12. Streamlit Dashboard

Run:
python -m streamlit run tokens_dashboard.py

yaml
Copy code

Shows:
- Users  
- Token balances  
- Basic statistics  

---

## 13. Testing the API (Swagger)

Recommended flow:

1. Sign up  
2. Log in  
3. Authorize  
4. Add tokens  
5. Train model  
6. List models  
7. Predict  

---

## 14. Future Improvements

- Add more ML models  
- Hyperparameter tuning  
- Better error handling  
- Enhanced dashboard  
- Docker support  
- Model comparison tools  
- Support for custom datasets  

---

## 15. Known Limitations

- Supports only the predefined dataset schema  
- Uses default hyperparameters unless provided  
- No caching or parallel training  
- Not fully production-ready  
