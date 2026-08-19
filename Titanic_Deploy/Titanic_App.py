"""
app.py — FastAPI prediction API for the Titanic Survival Prediction model.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from predict import predict_survival

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predicts whether a Titanic passenger would have survived.",
    version="1.0.0"
)


class PassengerInput(BaseModel):
    Pclass: Literal[1, 2, 3] = Field(..., description="Ticket class (1=1st, 2=2nd, 3=3rd)")
    Sex: Literal["male", "female"] = Field(..., description="Passenger sex")
    # Fixed: ge=0 allows infants under 1 year old
    Age: float = Field(..., ge=0, le=100, description="Passenger age in years")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0, description="Ticket fare paid")
    Title: Literal["Mr", "Mrs", "Miss", "Master", "Officer", "Royalty"] = Field(
        ..., description="Title derived from passenger name"
    )
    CabinKnown: bool = Field(..., description="Whether the passenger's cabin/deck is known")
    # Fixed: default=1 ensures omitted TicketFreq in raw calls defaults correctly
    TicketFreq: int = Field(default=1, ge=1, description="Number of passengers sharing this ticket")

    class Config:
        json_schema_extra = {
            "example": {
                "Pclass": 1,
                "Sex": "female",
                "Age": 38,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 71.28,
                "Title": "Mrs",
                "CabinKnown": True,
                "TicketFreq": 2
            }
        }


class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    survival_probability: float


@app.get("/")
def root():
    return {
        "message": "Titanic Survival Prediction API is running.",
        "docs": "/docs",
        "predict_endpoint": "/predict"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    try:
        result = predict_survival(passenger.model_dump())
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")