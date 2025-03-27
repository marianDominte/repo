from fastapi import APIRouter, HTTPException
from app.models import SymptomInput, DiseasePrediction
from app.services import predict_diseases

router = APIRouter(prefix="/api", tags=["Symptom Checker"])


@router.post("/predict", response_model=list[DiseasePrediction])
async def predict_symptoms(input_data: SymptomInput):
    """
    Endpoint pentru a face predicții pe baza simptomelor furnizate.
    """
    try:
        predictions = predict_diseases(input_data.symptoms)
        return predictions
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
