
1) Overview
This project implements an Automated Resume Screening System using Machine Learning and NLP to help recruiters efficiently classify resumes. It reduces manual effort, improves consistency, and supports fair candidate evaluation using a real-world Kaggle resume dataset.

2) Objectives
Preprocess and vectorize resume text using TF-IDF
Train multiple ML models for resume classification
Deploy a FastAPI-based REST API for real-time predictions
Provide a reproducible ML pipeline with clear documentation

3) Models Used
Logistic Regression
Support Vector Machine (SVM)
Random Forest

4)  How to Run
Setup Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Train & Evaluate Models
python3 -m src.train
python3 -m src.evaluate
Run FastAPI Server
uvicorn api.app:app --reload
Access API:
http://127.0.0.1:8000
Swagger UI: http://127.0.0.1:8000/docs


5) Project Structure
api/        FastAPI application
data/       Raw and processed datasets
src/        Preprocessing, training, evaluation scripts
models/     Trained models & TF-IDF vectorizer
reports/    Project documentation
slides/     Presentation

6) Results
All models achieved high classification performance on the test set:
Model	Accuracy	F1-score
Logistic Regression	1.00	1.00
SVM	1.00	1.00
Random Forest	1.00	1.00


7) Future Work
Add model explainability (SHAP/LIME)
Support PDF/DOCX resume uploads
Dockerize and add CI/CD pipeline


8) Conclusion
This project demonstrates an end-to-end ML-powered resume screening system using real-world data and FastAPI deployment, showcasing the practical application of AI in modern recruitment workflows.
 
