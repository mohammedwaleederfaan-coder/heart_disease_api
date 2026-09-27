# Heart Disease Prediction API

A machine learning API that predicts the likelihood of heart disease based on a patient's clinical measurements. Built with scikit-learn, served via FastAPI, and containerized with Docker.

## Live Docker Image

```bash
docker pull mohammedwaleederfaan/heart-disease-api
docker run -p 8000:8000 mohammedwaleederfaan/heart-disease-api
```

Once running, open `http://127.0.0.1:8000/docs` for the interactive API documentation (Swagger UI).

## Overview

This project follows a complete, production-style ML workflow:

1. **Explore & Clean** — Checked for duplicates, missing values, and outliers on a real clinical dataset; ran distribution and correlation analysis to understand feature relationships with the target.
2. **Preprocess** — Built a `ColumnTransformer` pipeline combining `StandardScaler` (numeric features) and `OneHotEncoder` (categorical features).
3. **Model & Select** — Benchmarked three classifiers (Logistic Regression, SVM, Random Forest) using 5-fold cross-validation, selecting the best model by **F1-score** — a more reliable metric than accuracy for medical data, where class imbalance can make accuracy misleading.
4. **Serve** — Wrapped the winning model in a FastAPI REST endpoint with Pydantic-based input validation, structured logging, and error handling.
5. **Containerize & Publish** — Packaged the app with Docker and published the image publicly on Docker Hub.

## Tech Stack

| Component | Technology |
|---|---|
| Modeling | scikit-learn — Logistic Regression, SVM, Random Forest |
| Preprocessing | ColumnTransformer (StandardScaler + OneHotEncoder) |
| Evaluation | Cross-validation, F1-score, Confusion Matrix |
| API Framework | FastAPI + Pydantic |
| Server | Uvicorn |
| Containerization | Docker |
| Registry | Docker Hub |

## Project Structure

```
.
├── main.py             # FastAPI application and prediction endpoint
├── model.pkl            # Trained classification pipeline (preprocessing + model)
├── model.ipynb           # EDA, preprocessing, model comparison and training
├── requirements.txt      # Python dependencies
└── Dockerfile            # Container build instructions
```

## Why F1-score, not Accuracy?

In medical prediction tasks, a model can score a high accuracy while still failing to catch actual disease cases — for example, if the dataset is imbalanced, a model that mostly predicts "no disease" can look accurate while being clinically useless. F1-score balances:
- **Recall** — of all patients who actually have the disease, how many did the model correctly identify?
- **Precision** — of all patients the model flagged as having the disease, how many actually did?

This gives a far more trustworthy picture of real-world model performance for this kind of task.

## API Endpoints

### `GET /`
Health check endpoint to confirm the API is running.

**Response:**
```json
{ "message": "Server Running" }
```

### `POST /predict`
Predicts whether a patient is likely to have heart disease based on clinical features.

**Request body:**
```json
{
  "Age": 52,
  "Sex": "M",
  "ChestPainType": "ATA",
  "RestingBP": 130,
  "Cholesterol": 246,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 173,
  "ExerciseAngina": "N",
  "Oldpeak": 0.0,
  "ST_Slope": "Up"
}
```

**Response:**
```json
{
  "prediction": 0
}
```
(`0` = no heart disease predicted, `1` = heart disease predicted)

## Running Locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then visit `http://127.0.0.1:8000/docs`.

## Running with Docker (build it yourself)

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

## Model Performance

Multiple classifiers were compared using 5-fold cross-validation, with the final model selected based on F1-score on a held-out test set to account for potential class imbalance in the clinical data.

## What This Project Demonstrates

- End-to-end ML workflow on real clinical data: EDA, cleaning, preprocessing, model comparison, and evaluation
- Choosing the right evaluation metric for the problem domain (F1-score over accuracy for imbalanced medical classification)
- Wrapping a trained ML pipeline in a production-style REST API with input validation and structured logging
- Containerizing and publishing a reproducible, one-command-deployable image to Docker Hub

## Author

Mohammed Waleed
[GitHub](https://github.com/mohammedwaleederfaan-coder/heart-disease-api) · [Docker Hub](https://hub.docker.com/r/mohammedwaleederfaan/heart-disease-api)
