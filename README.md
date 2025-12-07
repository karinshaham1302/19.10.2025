# Machine Learning & FastAPI Final Project

This project implements a complete machine-learning workflow using FastAPI, including dataset upload, model training, prediction, authentication, token management, and a Streamlit dashboard for monitoring usage.

---

## 1. Project Structure

19.10.2025/
│
├── app/
│ ├── routers/
│ │ ├── auth.py
│ │ ├── training.py
│ │ └── prediction.py
│ │
│ ├── models/ # Stored trained ML models (.pkl)
│ ├── logs/ # Application logs (optional)
│ │
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

## 2. Environment Setup

### 2.1 Create virtual environment

python -m venv .venv

shell
Copy code

### 2.2 Activate virtual environment (Windows)

.venv\Scripts\activate

shell
Copy code

### 2.3 Install project dependencies

pip install -r requirements.txt

yaml
Copy code

---

## 3. Running the FastAPI Server

Start the API server:

uvicorn app.main:app --reload

arduino
Copy code

Swagger UI is available at:

http://127.0.0.1:8000/docs

yaml
Copy code

---

## 4. Authentication (JWT)

### 4.1 Sign up

POST /auth/signup

css
Copy code

Example body:
```json
{
  "username": "user1",
  "password": "pass1234"
}
4.2 Log in and receive JWT
bash
Copy code
POST /auth/login
Example body:

json
Copy code
{
  "username": "user1",
  "password": "pass1234"
}
Response:

json
Copy code
{
  "access_token": "<JWT_TOKEN>"
}
4.3 Authorizing in Swagger
In Swagger UI, click Authorize and paste only the token (without the word "Bearer").

5. Token System
Training a model consumes 1 token

Making a prediction consumes 5 tokens

5.1 Check remaining tokens
bash
Copy code
GET /auth/tokens
5.2 Add tokens
bash
Copy code
POST /auth/add_tokens
Body:

json
Copy code
{
  "amount": 20
}
6. Model Training
Endpoint:

bash
Copy code
POST /training/train
Form-data fields:

file: CSV dataset

model_name: linear / decision_tree / random_forest

model_params: optional JSON

Response includes metrics:

r2

mae

mse

rmse

And saved model path + metadata.

7. Making Predictions
Endpoint:

bash
Copy code
POST /models/predict/{model_name}
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
Prediction is returned rounded to two decimal places.

8. Streamlit Dashboard
Run the dashboard:

arduino
Copy code
python -m streamlit run tokens_dashboard.py
The dashboard displays:

All users

Remaining tokens per user

9. Future Improvements
Add advanced ML algorithms (XGBoost, SVM, Gradient Boosting).

Add batch prediction endpoint.

Add role-based access (Admin vs User).

Add a frontend UI for model training and prediction.

Add email notifications for low token balance.

Add Docker support for deployment.

Add automatic feature/label inference from datasets.

10. Notes
The dataset used in this project is private_lessons_data.csv, stored under /data.

Model predictions and metrics remain consistent because the schema is predefined.

Authentication uses JWT for secure access to protected endpoints.


