
from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_model

# Load models and vectorizer
logistic_model = load_model("logistic_regression.pkl")
svm_model = load_model("svm.pkl")
rf_model = load_model("random_forest.pkl")
vectorizer = load_model("vectorizer.pkl")

# FastAPI app
app = FastAPI(title="Resume Screening API")

class ResumeInput(BaseModel):
    resume_text: str

@app.post("/predict")
def predict_resume(input: ResumeInput):
    X = vectorizer.transform([input.resume_text])
    return {
        "logistic_regression": logistic_model.predict(X)[0],
        "svm": svm_model.predict(X)[0],
        "random_forest": rf_model.predict(X)[0]
    }
