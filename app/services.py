import os
import numpy as np
from typing import List
from app.logistic_regression import prepare_data, predict, train_logistic_regression
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# Ruta fișierului CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "diseases.csv")


def load_model_and_data():
    """Încarcă datele și antrenează modelul logistic regression."""
    # Citim fișierul CSV
    df = pd.read_csv(CSV_PATH)

    # Prelucrăm datele
    X, y = prepare_data(df)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Preprocesarea datelor: adăugăm o coloană de bias
    X = np.c_[np.ones(X.shape[0]), X]
    num_classes = len(np.unique(y))

    # Antrenăm modelul logistic regression
    all_theta = train_logistic_regression(X, y, num_classes)

    return all_theta, label_encoder, df.columns[:-1]


# Încarcă modelul la pornirea aplicației
MODEL, LABEL_ENCODER, SYMPTOMS = load_model_and_data()


def predict_diseases(input_symptoms: List[str]) -> List[dict]:
    """Returnează lista cu bolile și probabilitățile bazate pe simptome."""
    # Validăm simptomele introduse
    symptom_vector = [1 if symptom.strip() in SYMPTOMS else 0 for symptom in SYMPTOMS]

    if not any(symptom_vector):
        raise ValueError("Niciun simptom valid nu a fost furnizat.")

    # Adăugăm termenul de bias
    user_vector = np.array(symptom_vector).reshape(1, -1)
    user_vector = np.c_[np.ones(user_vector.shape[0]), user_vector]

    # Calculăm probabilitățile
    probabilities = predict(MODEL, user_vector)

    # Sortăm rezultatele și luăm primele 10 boli
    sorted_indices = np.argsort(probabilities[0])[::-1]
    results = [
        {"disease": LABEL_ENCODER.classes_[idx], "probability": probabilities[0][idx] * 100}
        for idx in sorted_indices[:10]
    ]

    return results
