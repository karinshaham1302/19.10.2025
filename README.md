# Machine Learning & FastAPI Final Project

This project implements a complete machine learning pipeline served through a FastAPI backend.  
It supports training regression models on a predefined private-lessons dataset and making predictions through authenticated API calls using JWT tokens.  
The system also includes a token-based usage mechanism and an admin dashboard (Streamlit).

---

## 1. TL;DR — Short Version

- Upload dataset → Train ML model (Linear, Decision Tree, Random Forest).  
- System validates schema automatically (fixed dataset structure).  
- Model is saved with full metadata (features, label, metrics).  
- JWT authentication required.  
- Each API action consumes user tokens.  
- Prediction is performed using the latest trained model.  
- Admin dashboard (Streamlit) displays all users and their token balance.

---

## 2. How It Works — System Flow

1. **User registers** (`/auth/signup`)  
2. **User logs in** and receives a **JWT access token** (`/auth/login`)  
3. User uploads `private_lessons_data.csv` to **train a model** (`/training/train`)  
4. Model is trained, evaluated, saved to `/models/` and logged in metadata  
5. User requests **prediction** (`/models/predict/{model_name}`)  
6. System loads the latest trained model and returns a predicted price  
7. All actions consume tokens (predict = 5 tokens)

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
│   └── (saved .pkl model files)
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md
4. Dataset
The system works with one fixed dataset, stored as:

bash
Copy code
data/private_lessons_data.csv
Columns include:

subject

student_level

lesson_minutes

teacher_experience_years

is_online

city

teacher_age

lesson_price (label)

All training and prediction logic assumes this schema.

5. Running the Project
Step 1 — Create & Activate Virtual Environment
bash
Copy code
python -m venv .venv
source .venv/bin/activate     # Mac/Linux
.venv\Scripts\activate        # Windows
Step 2 — Install Dependencies
bash
Copy code
pip install -r requirements.txt
Step 3 — Start FastAPI Server
bash
Copy code
uvicorn app.main:app --reload
Step 4 — Open API Documentation
arduino
Copy code
http://127.0.0.1:8000/docs
6. Authentication Flow (JWT)
1. Register
POST /auth/signup

json
Copy code
{
  "username": "user1",
  "password": "pass1234"
}
2. Login
POST /auth/login

Returns:

json
Copy code
{
  "access_token": "<TOKEN>",
  "token_type": "bearer"
}
3. Add Token in Swagger
Click Authorize → Insert only the token (no “Bearer”).

7. Training a Model
Endpoint:

bash
Copy code
POST /training/train
Parameters (form-data):

file = private_lessons_data.csv

model_name = linear

model_params = {}

Expected Response:

json
Copy code
{
  "status": "success",
  "message": "Model was trained successfully and is ready for predictions.",
  "model_info": {
    "model_name": "linear",
    "n_samples": 100,
    "metrics": {
      "r2": 0.96,
      "mae": 5.17,
      "rmse": 7.43
    }
  }
}
8. Listing All Models
bash
Copy code
GET /models
Returns all trained models with metrics.

9. Making Predictions
Endpoint:

bash
Copy code
POST /models/predict/linear
Body:

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
10. Streamlit Admin Dashboard
Run:

bash
Copy code
python -m streamlit run tokens_dashboard.py
Shows:

all users

token balances

total token statistics

11. Future Improvements
Add more ML algorithms (SVR, XGBoost).

Add feature scaling & preprocessing selection.

Add database migration (Alembic).

Streamlit interface for training and prediction.

User roles (admin / user).

Docker containerization.

12. Notes
Dataset is fixed and schema-dependent.

JWT tokens must be renewed after expiration.

Models are saved automatically in /models/ with metadata.

Errors are logged to app/logs/server.log.



