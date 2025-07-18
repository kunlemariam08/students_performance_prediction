from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from predict import predict_graduation_class  

# Load model, scaler, and feature columns
try:
    model = joblib.load("student_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except Exception as e:
    print(f"Error loading model files: {e}")
    raise

# FastAPI app
app = FastAPI(title="Student Graduation Prediction API", version="1.0.0")

# Pydantic input schema
class StudentInput(BaseModel):
    age: int
    gender: int
    ethnicity: int
    parentaleducation: int
    studytimeweekly: float
    absences: int
    tutoring: int
    parentalsupport: int
    extracurricular: int
    sports: int
    music: int
    volunteering: int

@app.get("/")
def root():
    return {"message": "Student Graduation Prediction API is running"}


@app.post("/predict")
def predict_graduation(student: StudentInput):
    try:
        # Convert Pydantic model to dictionary
        student_dict = student.dict()
        
        # Predict using your prediction function
        result = predict_graduation_class(model, scaler, feature_columns, student_dict)
        
        return {
            "input": student_dict,
            "prediction": result,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running properly"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)