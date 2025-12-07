# 🧠 Machine Learning & FastAPI Final Project

FastAPI-based backend system for training machine learning models, managing users with JWT authentication, handling token-based usage limitations, and performing predictions using trained models.

This project demonstrates a complete ML pipeline with a production-style API, including authentication, model storage, metadata tracking, token economy, logging, and an optional Streamlit dashboard.

---

## 🚀 Features

### 🔐 **Authentication & Authorization**
- User signup & login using **JWT tokens**
- Secure endpoints with `Bearer <token>`
- Token-based usage limits:
  - **Training** → 1 token  
  - **Prediction** → 5 tokens  

### 🤖 **Machine Learning Capabilities**
- Train models from CSV files  
- Supported models:
  - Linear Regression  
  - Decision Tree  
  - Random Forest  
- Automatic preprocessing (OneHotEncoder + numeric passthrough)
- Saves model + metadata into `/app/models/`
- Evaluation metrics:
  - R², MAE, MSE, RMSE (rounded to 2 decimals)

### 📊 **Prediction API**
- Predict with the **latest trained model**  
- Input via JSON `{ "data": { ... } }`
- Validates required features automatically

### 📁 **Model Metadata Tracking**
Stored in `models_metadata.json`:
- model_id  
- model_name  
- model_type  
- trained_at  
- features used  
- label column  
- metrics  
- model_path  

### 💳 **Token Economy**
- `/auth/tokens` — check balance  
- `/auth/add_tokens` — add tokens  
- `/auth/remove_user` — delete user  

### 📉 **Streamlit Dashboard (Optional)**
`tokes_dashboard.py` displays:
- All users  
- Remaining tokens  

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **FastAPI** | REST API server |
| **Scikit-learn** | Training ML models |
| **Pandas** | Dataset manipulation |
| **Passlib (bcrypt)** | Password hashing |
| **SQLite** | User & token storage |
| **JWT (PyJWT)** | Authentication |
| **Uvicorn** | ASGI server |
| **Streamlit** | Optional dashboard |

---

## 📂 Project Structure

19.10.2025/
│
├── app/
│ ├── routers/
│ │ ├── auth.py
│ │ ├── training.py
│ │ └── prediction.py
│ │
│ ├── models/ # Saved .pkl models
│ ├── logs/ # Log files (if generated)
│ ├── init.py
│ ├── auth_service.py
│ ├── model_service.py
│ ├── database.py
│ ├── schemas.py
│ ├── config.py
│ └── main.py
│
├── data/
│ └── private_lessons_data.csv
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md

yaml
Copy code

---

## 📦 Installation

### 1️⃣ Create & activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run FastAPI server
bash
Copy code
uvicorn app.main:app --reload
4️⃣ Open Swagger UI
arduino
Copy code
http://127.0.0.1:8000/docs
🔐 Authentication Flow
⭐ Signup
POST /auth/signup

json
Copy code
{
  "username": "user1",
  "password": "pass1234"
}
⭐ Login
POST /auth/login
Copy the access_token from the response.

⭐ Add authentication to Swagger
Click “Authorize” → paste ONLY the token (no need to write "Bearer").

📊 Model Training
Endpoint
POST /training/train

Example form-data:
ini
Copy code
file = private_lessons_data.csv
model_name = linear
model_params = {"fit_intercept": true}
Response example:

json
Copy code
{
  "status": "success",
  "message": "Model was trained successfully and is ready for predictions.",
  "model_info": {
    "model_id": 2,
    "model_name": "linear",
    "r2": 0.96,
    "mae": 5.17,
    "mse": 52.83,
    "rmse": 7.27
  }
}
🎯 Predictions
Endpoint
POST /models/predict/{model_name}

Example body
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
Example response
json
Copy code
{
  "model_name": "linear",
  "model_id": 2,
  "prediction": 163.04
}
🔍 Check Available Models
GET /models/

Output:

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
      "mse": 52.83
    }
  ]
}
🧮 Token System
Check tokens
GET /auth/tokens

Add tokens
POST /auth/add_tokens

json
Copy code
{
  "amount": 20
}
Delete user
DELETE /auth/remove_user

json
Copy code
{
  "username": "user1",
  "password": "pass1234"
}
🧰 Optional: Streamlit Dashboard
Run:

bash
Copy code
python -m streamlit run tokens_dashboard.py
Shows:

All users

Tokens remaining

🚀 Future Improvements
These enhancements can elevate the project to production-level quality:

✔ Add Logistic Regression, SVM, XGBoost
Expands model capabilities and allows classification tasks.

✔ Add ML model versioning system
Choose model version instead of “latest only”.

✔ Add role-based permissions
Admin vs standard users.

✔ Add Docker deployment
Package the API to run anywhere.

✔ Add CI/CD pipeline
Automatic testing before every push.

📝 Notes
All metrics and predictions are rounded to 2 decimal places.

JWT tokens must be reissued after expiration.

CSV structure is fixed to the private lessons dataset for the project.

🎉 Final Words
This project demonstrates:

Full backend engineering

Machine learning integration

Secure authentication

Deployment-ready API structure

A complete, professional final project.
