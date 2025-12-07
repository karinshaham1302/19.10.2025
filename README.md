# Machine Learning & FastAPI — Final Project

This project implements an end-to-end machine learning system for predicting private-lesson prices using:
FastAPI, scikit-learn, JWT authentication, a token-based usage system, and a Streamlit admin dashboard.

---

## 1. TL;DR (Short Summary)

- Fixed dataset: `data/private_lessons_data.csv`
- Train ML models: **Linear Regression**, **Decision Tree**, **Random Forest**
- Make predictions through authenticated API calls
- JWT-based authentication with per-user token balance
- Models saved as `.pkl` files with full metadata and metrics (JSON)
- Streamlit dashboard shows users and token balances

---

## 2. System Flow (High-Level)

1. User signs up → `POST /auth/signup`
2. User logs in → receives JWT token → `POST /auth/login`
3. User uploads dataset and trains a model → `POST /training/train`
4. Model is trained, evaluated, and saved under `models/` (+ JSON metadata)
5. User sends a prediction request → `POST /models/predict/{model_name}`
6. System loads the latest model and returns the predicted price
7. Each action consumes tokens according to configured costs
8. Admin can view all users and tokens via the Streamlit dashboard

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
│   ├── (saved .pkl model files)
│   └── models_metadata.json
│
├── project_info.ipynb
├── tokens_dashboard.py
├── requirements.txt
└── README.md
## 4. Dataset and Notebook

### 4.1 Dataset: `private_lessons_data.csv`

This is a fixed dataset of private lessons used by both the API and the notebook.

#### **Schema**

| Column                   | Type   |
|--------------------------|--------|
| subject                  | string |
| student_level            | string |
| lesson_minutes           | int    |
| teacher_experience_years | int    |
| is_online                | string |
| city                     | string |
| teacher_age              | int    |
| lesson_price (label)     | float  |

#### **Feature columns used for training**
```python
["subject", "student_level", "lesson_minutes",
 "teacher_experience_years", "is_online", "city"]
### 4.2 Jupyter Notebook: `project_info.ipynb`

The notebook includes:

- Loading the dataset with **pandas**
  - `df.head()`, `df.info()`, `df.describe(include="all")`
- **Exploratory Data Analysis (EDA)**:
  - Price distribution
  - Lesson duration distribution
  - Teacher experience distribution
  - Breakdowns by **subject**, **student level**, **city**, **online/offline**
  - Correlation heatmap for numeric features
- **Simple Linear Regression training example**
- **Evaluation metrics** (rounded to 2 decimals):
  - R²  
  - MAE  
  - RMSE
- A summary of key insights

The notebook serves as a readable explanation of the same dataset used by the API.
## 5. Technologies

| Component        | Technology              |
|------------------|--------------------------|
| API Framework    | FastAPI                  |
| Authentication   | JWT (python-jose)        |
| Password Hashing | bcrypt                   |
| ML Engine        | scikit-learn             |
| Data Handling    | pandas                   |
| Model Storage    | joblib + JSON metadata   |
| Database         | SQLite                   |
| Dashboard        | Streamlit                |
| Python Version   | 3.x                      |
```markdown
## 6. Installation

### 6.1 Clone the Repository
git clone https://github.com/karinshaham1302/19.10.2025.git
cd 19.10.2025

### 6.2 Create Virtual Environment
Windows:
python -m venv .venv
.venv\Scripts\activate

macOS / Linux:
python -m venv .venv
source .venv/bin/activate

### 6.3 Install Dependencies
pip install -r requirements.txt
```
```markdown
## 7. Running the FastAPI Server

### 7.1 Start the Server
Run from the project root (19.10.2025/) with the virtual environment activated:

uvicorn app.main:app --reload

### 7.2 Access the API
- API Root: http://127.0.0.1:8000/
- Swagger UI (interactive documentation): http://127.0.0.1:8000/docs

Swagger UI allows:
- Executing all API endpoints
- Uploading files for model training
- Testing authentication
- Sending prediction requests

This is the recommended interface for testing the project.
```
```markdown
## 8. Authentication & Token System

The project uses **JWT authentication** and a **token-based usage system**.  
Every protected endpoint requires a valid JWT token, and certain operations deduct tokens.

---

### 8.1 Token Costs (from `config.py`)
| Action             | Tokens Required |
|--------------------|-----------------|
| Train model        | 1               |
| Train multiple     | 1               |
| Predict            | 5               |

---

### 8.2 Authentication Flow

1. **Sign Up**  
   `POST /auth/signup`  
   Creates a new user.

2. **Log In**  
   `POST /auth/login`  
   Returns an `access_token` (JWT).

3. **Authorize in Swagger**  
   - Open: http://127.0.0.1:8000/docs  
   - Click **Authorize**
   - Paste **only** the token (not the word "Bearer")

4. All protected routes now authenticate automatically.

---

### 8.3 Main Authentication Endpoints

| Method | Endpoint           | Description                           | Auth Required |
|--------|--------------------|----------------------------------------|---------------|
| POST   | /auth/signup       | Register a new user                    | No            |
| POST   | /auth/login        | Log in and receive JWT token           | No            |
| GET    | /auth/tokens       | Get current token balance              | Yes (JWT)     |
| POST   | /auth/add_tokens   | Add tokens to the logged-in user       | Yes (JWT)     |
| DELETE | /auth/remove_user  | Delete a user by username + password   | No (body)     |

---
```
```markdown
## 9. Model Service (Machine Learning Logic)

All machine-learning functionality is implemented inside:
`app/model_service.py`

This module is responsible for:
- Validating the dataset structure  
- Building preprocessing pipelines  
- Creating ML models (Linear, Decision Tree, Random Forest)  
- Training and evaluating models  
- Saving trained models + metadata  
- Loading models for prediction  

---

### 9.1 Dataset Validation

Before training, the service checks that **all required columns** exist.

Required columns:

| Column                     | Role    |
|---------------------------|---------|
| subject                   | Feature |
| student_level             | Feature |
| lesson_minutes            | Feature |
| teacher_experience_years  | Feature |
| is_online                 | Feature |
| city                      | Feature |
| lesson_price              | Label   |

If any column is missing → an error is raised.

---

### 9.2 Preprocessing Pipeline

Implemented using `ColumnTransformer`:

- **Categorical columns** → OneHotEncoder(handle_unknown="ignore")
- **Numeric columns** → passthrough

This ensures:
- Categorical values are encoded correctly  
- Unknown categories during prediction do not break the pipeline  

---

### 9.3 Supported Model Types

| Model Name       | Algorithm Used                |
|------------------|-------------------------------|
| `"linear"`       | LinearRegression              |
| `"decision_tree"`| DecisionTreeRegressor        |
| `"random_forest"`| RandomForestRegressor        |

Each model is wrapped in a **Pipeline** that includes preprocessing.

---

### 9.4 Training Process (`train_model()`)

Steps performed:

1. Validate dataset  
2. Split into train/test (`test_size=0.2`)  
3. Build preprocessing  
4. Build selected model  
5. Fit model on training set  
6. Predict on test set  
7. Calculate metrics (rounded to 2 decimals):

| Metric | Meaning |
|--------|---------|
| R²     | Model accuracy (explained variance) |
| MAE    | Mean Absolute Error |
| MSE    | Mean Squared Error |
| RMSE   | Root Mean Square Error |

Example saved metrics:
```json
{
  "r2": 0.96,
  "mae": 5.17,
  "mse": 52.83,
  "rmse": 7.27
}
```

---

### 9.5 Saving the Model

Each trained model is saved as a `.pkl` file inside:

```
/models/
```

Metadata is appended to:

```
/models/models_metadata.json
```

Metadata includes:

- model_id  
- model_name  
- timestamp  
- features used  
- label column  
- metrics  
- absolute model_path  

---

### 9.6 Utility Functions

| Function                         | Purpose                                    |
|----------------------------------|---------------------------------------------|
| `get_all_models()`               | Returns full list of all model records      |
| `get_latest_model_record(name)`  | Returns the **newest trained model**        |
| `load_model_from_record(record)` | Loads a trained `.pkl` pipeline from disk   |

These enable both the **training API** and the **prediction API** to work seamlessly.

---
```
```markdown
## 10. Training API

All training-related endpoints are defined in:
`app/routers/training.py`

The training API allows uploading the dataset, choosing a model type, and saving the trained model + metadata for later predictions.

---

### 10.1 Endpoint: **Train Single Model**
**POST `/training/train`**

This endpoint trains **one model** using a CSV file (e.g., `private_lessons_data.csv`).

#### Request Type
`multipart/form-data`

#### Required Fields
| Field        | Type     | Description |
|--------------|----------|-------------|
| `file`       | CSV file | The dataset to train on |
| `model_name` | string   | `"linear"`, `"decision_tree"`, `"random_forest"` |
| `model_params` | JSON string (optional) | Model hyperparameters (e.g. `{"max_depth": 5}`) |

#### Requirements
- Valid **JWT token**  
- User must have **at least 1 token** (`TRAIN_TOKENS_COST = 1`)

#### What Happens Internally
1. CSV is read into a DataFrame  
2. Dataset columns are validated  
3. Preprocessing pipeline is built  
4. Selected model is trained  
5. Model is evaluated (R², MAE, MSE, RMSE)  
6. Model is saved to `/models/`  
7. Metadata is appended to `models_metadata.json`  

#### Example Successful Response
```json
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
    "model_path": "models/linear_20251207_101812.pkl"
  }
}
```

---

### 10.2 Endpoint: **Train Multiple Models**
**POST `/training/train_multi`**

This endpoint trains **several model types** in one request.

#### Fields
| Field          | Type  | Description |
|----------------|-------|-------------|
| `file`         | CSV   | Dataset |
| `model_names`  | JSON list | Example: `["linear", "random_forest"]` |
| `model_params` | JSON (optional) | Hyperparameters |

#### Behavior
- Trains each model independently  
- Saves each trained model  
- Returns metrics per model  
- Reports which model achieved **best R² score**

#### Token Usage
- Costs **1 token total**, not per model (`TRAIN_MULTI_TOKENS_COST = 1`)

---

### 10.3 Returned Metadata (for all training endpoints)

| Field             | Description |
|-------------------|-------------|
| `model_id`        | Unique ID for the model |
| `model_name`      | User-selected name |
| `model_type`      | Actual scikit-learn class |
| `trained_at`      | UTC timestamp |
| `features_used`   | Columns used for training |
| `label_column`    | Target column |
| `n_samples`       | Dataset size |
| `metrics`         | R², MAE, MSE, RMSE |
| `model_path`      | Filepath of saved `.pkl` model |

This metadata allows:
- Re-loading models for prediction  
- Tracking training history  
- Listing trained models in the UI  

---
```
```markdown
## 11. Prediction API

All prediction-related endpoints are defined in:
`app/routers/prediction.py`

Predictions are always made using the **latest trained model** of the requested type (e.g., the newest `"linear"` model).

---

### 11.1 Endpoint: **List All Trained Models**
**GET `/models/`**

Returns a list of all models saved in `models_metadata.json`.

#### Requirements
- Valid **JWT token**

#### Response Contains
| Field        | Meaning |
|--------------|---------|
| `model_id`   | Unique ID of the model |
| `model_name` | The name used during training |
| `model_type` | Actual scikit-learn estimator class |
| `trained_at` | Timestamp (UTC) |
| `r2`, `mae`, `mse`, `rmse` | Evaluation metrics |

#### Example Response
```json
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
```

---

### 11.2 Endpoint: **Predict With Latest Model**
**POST `/models/predict/{model_name}`**

Example path:
```
/models/predict/linear
```

#### Requirements
- Valid **JWT token**
- User must have **at least 5 tokens** (`PREDICT_TOKENS_COST = 5`)

---

### Request Body
Wrap all feature values under `"data"`:

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
```

All columns **must match the dataset schema** exactly.

---

### What Happens Internally
1. The system loads the **most recent model** of the given type  
2. Converts `"data"` into a 1-row pandas DataFrame  
3. Applies the full preprocessing + model pipeline  
4. Returns the predicted price  
5. Deducts 5 tokens from the user  

---

### Example Successful Response
```json
{
  "model_name": "linear",
  "model_id": 2,
  "prediction": 163.04
}
```

Prediction is rounded to **two decimal places** in the API.

---

### Common Prediction Errors & Meaning

| Error Message | Meaning |
|---------------|---------|
| `Not authenticated` | JWT token missing or expired |
| `Not enough tokens` | User has fewer than 5 tokens |
| `Model not found` | No trained model exists with that name |
| `columns are missing` | The JSON `data` object is missing required fields |
| `Invalid token` | JWT signature invalid / token not pasted correctly |

These are normal and expected API validations.

---
```
```markdown
## 12. Streamlit Admin Dashboard

File: `tokens_dashboard.py`

The project includes a simple **admin dashboard** built with **Streamlit**, allowing the system owner to view all users and their token balances.

---

### 12.1 Purpose & Features

The dashboard provides:

- A table of **all registered users**
- Each user’s **current token balance**
- Basic system statistics:
  - total users
  - total tokens in circulation

It reads data directly from:
```
data/app.db
```
(the same SQLite database used by FastAPI).

---

### 12.2 How to Run the Dashboard

Activate the virtual environment and run:

```bash
python -m streamlit run tokens_dashboard.py
```

This will open Streamlit automatically in your browser at:

```
http://localhost:8501
```

---

### 12.3 What You Will See

- A clean table showing:
  - `id`
  - `username`
  - `tokens`
  - `created_at` (if implemented)
- Summary statistics panel
- Auto-updating table (on refresh)
- No authentication needed (local admin-only use)

---

### 12.4 Internal Logic

The dashboard:

1. Connects to `app.db`
2. Executes a SQL query on the `users` table
3. Loads results into a pandas DataFrame
4. Displays the DataFrame using Streamlit’s UI components

No write operations occur — it is **read-only**, ensuring database integrity.

---
```
```markdown
## 13. Testing the API with Swagger (Recommended Workflow)

FastAPI automatically provides an interactive API tester called **Swagger UI**.

It is available at:

```
http://127.0.0.1:8000/docs
```

Swagger allows you to:
- send requests  
- upload files  
- authenticate with JWT  
- test training & prediction  
- view all API responses clearly  

Below is the **recommended full test flow** for verifying the project manually.

---

### 13.1 Start the FastAPI Server
Make sure your virtual environment is active, then run:

```bash
uvicorn app.main:app --reload
```

The API is now live.

---

### 13.2 Open Swagger UI
Navigate to:

```
http://127.0.0.1:8000/docs
```

You will see all endpoints grouped by:
- **auth**
- **training**
- **models**

---

### 13.3 Full Test Flow (Step-by-Step)

#### **1️⃣ Sign Up (Register a new user)**
Use:
```
POST /auth/signup
```
Send:
```json
{
  "username": "demo_user",
  "password": "abc1234"
}
```

A new user is created.

---

#### **2️⃣ Log In (Get JWT access token)**
Use:
```
POST /auth/login
```
Send the same credentials.

You will receive:
```json
{
  "access_token": "<JWT HERE>",
  "token_type": "bearer"
}
```

Copy the **access_token** (without the word "bearer").

---

#### **3️⃣ Authorize in Swagger**
At the top of Swagger UI:

1. Click **Authorize**
2. Paste only the token into the text box
3. Click **Authorize → Close**

Now all protected endpoints will work.

---

#### **4️⃣ Add Tokens to the User**
Training and predictions require tokens.

Use:
```
POST /auth/add_tokens
```

Example:
```json
{
  "amount": 20
}
```

You now have enough credits for training & prediction.

---

#### **5️⃣ Train a Model**
Use:
```
POST /training/train
```

Upload:
- `private_lessons_data.csv`

Fields:
- `model_name = linear`
- `model_params` → empty or `{}`

A successful response includes:
- model ID  
- saved model path  
- metrics (R², MAE, MSE, RMSE)  

---

#### **6️⃣ View All Trained Models**
Use:
```
GET /models/
```

This returns a list of all saved models with metrics:
- `r2`, `mae`, `mse`, `rmse`

---

#### **7️⃣ Make a Prediction**
Use:
```
POST /models/predict/linear
```

Example request:
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
```

Example response:
```json
{
  "model_name": "linear",
  "model_id": 2,
  "prediction": 163.04
}
```

(rounded to 2 decimals)

---

### 13.4 Swagger Testing Summary

| Step | Purpose |
|------|---------|
| Sign-up | Create user |
| Log-in | Get JWT token |
| Authorize | Unlock protected routes |
| Add tokens | Enable training/prediction |
| Train model | Create `.pkl` model + metadata |
| List models | Verify storage |
| Predict | Test the entire ML pipeline |

If all steps pass successfully — the system is functioning end-to-end.

---
```
## 14. Future Improvements

To further enhance the system and make it more flexible, scalable, and production-ready, the following improvements can be added:

### **14.1 Additional Model Types**
Add more machine-learning algorithms such as:
- XGBoost  
- Support Vector Regression (SVR)  
- Gradient Boosting  
- Neural networks (MLPRegressor)

This will allow experimenting with more powerful models and comparing performance.

---

### **14.2 Hyperparameter Tuning**
Integrate:
- GridSearchCV  
- RandomizedSearchCV  

This will enable automatic model optimization rather than using scikit-learn defaults.

---

### **14.3 Expanded Model Metadata**
Model metadata can include:
- Train/test sizes  
- Confusion matrices (for classification)  
- Feature importances  
- Training duration  

More metadata improves traceability and evaluation.

---

### **14.4 Improve Error Handling**
Enhance validation and user messages for:
- Missing columns  
- Wrong datatypes  
- Missing tokens  
- Invalid JWT tokens  

This will make the API more robust and user-friendly.

---

### **14.5 Streamlit Dashboard Enhancements**
Extend the dashboard to include:
- Training history for each user  
- Token usage over time  
- Visualization of model metrics  
- Ability to trigger training or delete models from UI  

---

### **14.6 Add Automated Tests**
Using `pytest` for:
- Auth flow  
- Token logic  
- ML training  
- Prediction accuracy  

Automated testing increases reliability and prevents regressions.

---

### **14.7 Docker Deployment**
Add a `Dockerfile` and `docker-compose.yml` to simplify:
- Deployment  
- Environment setup  
- Running API + Streamlit + DB together  

---

### **14.8 Model Comparison Tools**
Create an endpoint or dashboard that:
- Compares multiple models (R², MAE, RMSE)  
- Highlights the best model automatically  
- Shows performance graphs  

---

### **14.9 Support for Custom Datasets**
Allow users to:
- Upload CSV files with **any** schema  
- Choose feature columns manually  

This would make the system fully general-purpose instead of dataset-specific.

---
## 15. Known Limitations

- The system currently supports **only the predefined private-lessons dataset schema**.
- Models use **default scikit-learn hyperparameters** unless manually specified.
- No **parallel training**, caching, or performance optimization is implemented.
- The project is not yet configured for **production/cloud deployment** (no Docker, CI/CD, load balancing).
- Only **regression models** are supported (Linear, Decision Tree, Random Forest).
