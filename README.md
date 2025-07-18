# students_performance_prediction
This repository is focused on understanding and summarising the academic and behavioural differences between students across various grade classifications.  This model forecasts a student’s likely final grade classification, providing valuable insights for educators and institutions to support academic success through timely interventions.

# Dataset 
This document explores a dataset containing academic outcomes and attributes for approximately 3,000 students. Our objective is to understand and summarise the different behaviours and characteristics between students who performed well academically and those who did not, using the historical data available. The dataset was sourced from a higher education institution as part of a student performance evaluation challenge. Before cleaning, we had a dataset comprising 2392 entries and 15 features. After data cleaning, we arrived at a refined dataset with 2,392 records and 13 features.

# Data Wrangling
🔧 Data Preprocessing Steps
Convert all column names to lowercase
To keep the column names consistent and easier to work with.

Drop unimportant columns
Remove columns that do not contribute meaningfully to the prediction (e.g., student_id, address, or any irrelevant metadata).

Reduce decimal precision of numerical values
Round studytimeweekly and gpa to 2 decimal places to improve readability and reduce noise.

Create a new column for grade classification
Write a function that takes the GPA and assigns a grade category such as:

First Class

Second Class Upper

Second Class Lower

Third Class

Fail

Map GPA values to grade labels
Use defined GPA thresholds to classify students. 

# 📊 Summary of Findings
High GPA is linked to positive academic behaviours: Students with higher GPAs typically reported more weekly study time, higher attendance rates, and better access to learning resources.

Parental education and support matter: Students whose parents attained higher levels of education tended to perform better, suggesting the influence of home learning environments.

Low study time and irregular attendance are warning signs: Students with low weekly study hours and poor attendance were more likely to fall into lower grade categories.

Balanced lifestyle matters: Students who reported excessive free time or part-time jobs with long hours showed slightly lower academic performance, indicating the impact of time management.

GPA to Grade Mapping helped simplify prediction: Mapping GPA to clear grade categories (e.g., First Class, Second Class Upper, etc.) improved interpretability for both educators and predictive modeling.


# 🎯 Key Insights for Presentation
Study Time Strongly Predicts Academic Success
Students who dedicated more time to studying each week had a higher likelihood of graduating with First Class or Second Class Upper degrees.

Attendance Is a Critical Factor
Regular attendance was closely associated with better performance. Students with poor attendance were more likely to be in the lower grade categories.

Parental Education Influences Student Outcomes
Students whose parents had tertiary education tended to perform better academically, suggesting the role of family background in academic achievement.

Internet Access Supports Learning
Access to the internet at home correlated with higher grades, possibly due to better access to educational resources and online learning support.

Socioeconomic Indicators Matter
Students with financial challenges or those working long hours part-time showed lower academic performance, highlighting the importance of socioeconomic support.

GPA Can Be Effectively Mapped to Grade Categories
Using GPA to predict clear grade classifications (like First Class, Second Class Lower, etc.) made the model outputs more understandable and actionable.

Model Can Identify At-Risk Students Early
The predictive model can help educators and administrators identify students at risk of underperforming, allowing for timely academic interventions.

# Project Documentation: Student Performance Prediction API

## 📘 Overview

This project simulates a real-world AI workflow where a data scientist is responsible for building, validating, and deploying a machine learning model that predicts a student's graduation classification (e.g., First Class, Second Class Upper) based on academic engagement and performance features.

The model is exposed via a REST API (using FastAPI) and designed to be consumed by backend systems like a Laravel-based LMS or SMS.

---

## 📁 Project Structure

```
student-performance-predictor/
├── students_performance_data.csv
├── predict.py
├── feature_columns.pkl
├── student_model.pkl
├── feature_scaler.pkl
├── student_grade_classification.py
├── app.py
├── requirements.txt
```

---

## 📊 Dataset Description

A sample dataset with the following features was used:
  ----  
 0   StudentID          2392 non-null   int64  
 1   Age                2392 non-null   int64  
 2   Gender             2392 non-null   int64  
 3   Ethnicity          2392 non-null   int64  
 4   ParentalEducation  2392 non-null   int64  
 5   StudyTimeWeekly    2392 non-null   float64
 6   Absences           2392 non-null   int64  
 7   Tutoring           2392 non-null   int64  
 8   ParentalSupport    2392 non-null   int64  
 9   Extracurricular    2392 non-null   int64  
 10  Sports             2392 non-null   int64  
 11  Music              2392 non-null   int64  
 12  Volunteering       2392 non-null   int64  
 13  GPA                2392 non-null   float64
 14  GradeClass         2392 non-null   float64
dtypes: float64(3), int64(12)

---

# Document the API clearly, showing:

# The expected input and output format (e.g., JSON)

# A sample request/response

# Any libraries or tools used

# A brief guide for how a Laravel backend developer might call and use your endpoint ##


## 🤖 Model Training

File: train\_model.py

- Uses scikit-learn's RandomForestClassifier.
- Encodes the target class using LabelEncoder.
- Trains on 80% of the dataset, tests on 20%.
- Saves the model and label encoder with joblib.

---

## 🚀 API Deployment (FastAPI)

File: app.py

- Endpoint: POST /predict
- Accepts JSON input with all features.
- Returns the predicted degree classification.


### Example Output

```json
{
  "predicted_class": "Second Class Upper"
}
```
---

## 🧠 Future Enhancements

- Add input validation and error handling.
- Deploy API using Docker or cloud platforms (Render, Railway).
- Add model versioning and retraining workflows.
- Integrate with a monitoring dashboard for model performance.

---

## 📦 Requirements

Contents of requirements.txt:

```
fastapi
uvicorn
scikit-learn
pandas
joblib
```

---
