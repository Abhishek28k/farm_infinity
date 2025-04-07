import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from .forms import CropPredictionForm
import pandas as pd

model_path = os.path.join(settings.MEDIA_ROOT, 'model', 'crop_model.h5')
model = load_model(model_path)

class_names = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 
               'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 
               'papaya', 'coconut', 'cotton', 'jute', 'coffee']

csv_file_path = os.path.join(settings.MEDIA_ROOT,'dataset', 'Crop_recommendation.csv')
crop_data = pd.read_csv(csv_file_path)
features_data = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
scaler = StandardScaler().fit(features_data)

def predict_crop(request):
    accuracy = 0.0
    prediction = None
    if request.method == 'POST':
        form = CropPredictionForm(request.POST)
        if form.is_valid():
            features = np.array([
                form.cleaned_data['N'],
                form.cleaned_data['P'],
                form.cleaned_data['K'],
                form.cleaned_data['temperature'],
                form.cleaned_data['humidity'],
                form.cleaned_data['ph'],
                form.cleaned_data['rainfall']
            ]).reshape(1, -1)
            
            features_scaled = scaler.transform(features)
            result = model.predict(features_scaled)
            prediction_idx = np.argmax(result, axis=1)[0]
            prediction = class_names[prediction_idx]
            accuracy = np.max(result) * 100 
            print(result, prediction_idx, prediction)

    else:
        form = CropPredictionForm()

    return render(request, 'croppredictionapp/home.html', {'form': form, 'prediction': prediction, 'accuracy': accuracy})
