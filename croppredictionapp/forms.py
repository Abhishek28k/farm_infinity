# croppredictionapp/forms.py

from django import forms

class CropPredictionForm(forms.Form):
    N = forms.FloatField(label="Nitrogen (N)")
    P = forms.FloatField(label="Phosphorus (P)")
    K = forms.FloatField(label="Potassium (K)")
    temperature = forms.FloatField(label="Temperature (°C)")
    humidity = forms.FloatField(label="Humidity (%)")
    ph = forms.FloatField(label="pH Level")
    rainfall = forms.FloatField(label="Rainfall (mm)")
