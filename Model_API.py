# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Validation Class
class ModelFeatures(BaseModel):
    age: int
    gender: int
    bmi: float
    smoking: int
    alcohol_consumption: float
    physical_activity: float
    sleep_quality: float
    family_history_alzheimers: int
    cardiovascular_disease: int
    diebetes: int
    depression: int
    head_injury: int
    hypertension: int
    mmse: float
    functional_assessment: float
    memory_complaints: int
    behavioral_problems: int
    confusion: int
    disorientation: int
    personality_changes: int
    difficulty_completing_tasks: int
    forgetfulness: int
    
# Main app object
app = FastAPI()
model = keras.models.load_model("model/Alzheimers_Disease_Detection_Model_v1.0.h5")

# Routes
@app.get("/")
async def root():
    return {"message":"Alzheimers Disease Detection API"}

@app.post("/predict")
async def predict(features: ModelFeatures):
    try:
        # Convert Pydantic model to NumPy array
        data = np.array([[
            features.age,
            features.gender,
            features.bmi,
            features.smoking,
            features.alcohol_consumption,
            features.physical_activity,
            features.sleep_quality,
            features.family_history_alzheimers,
            features.cardiovascular_disease,
            features.diebetes,
            features.depression,
            features.head_injury,
            features.hypertension,
            features.mmse,
            features.functional_assessment,
            features.memory_complaints,
            features.behavioral_problems,
            features.confusion,
            features.disorientation,
            features.personality_changes,
            features.difficulty_completing_tasks,
            features.forgetfulness
        ]], dtype=np.float32)  # Ensure data type compatibility
        prediction = model.predict(data)[0][0]  # Extract scalar value from array
        probability = round(prediction * 100, 2)
        diagnosis = "Yes" if prediction > 0.5 else "No"
        return {"diagnosis": diagnosis, "probability": f"{probability}%"}
    except Exception as error:
        print(error)
        
@app.get("/version")
async def version():
    return {"version": "1.0"}