from fastapi import FastAPI
from app.routes import router

# Creăm instanța aplicației
app = FastAPI(title="Symptom Checker API")

# Înregistrăm rutele definite în routes.py
app.include_router(router)

# Rădăcina aplicației
@app.get("/")
async def root():
    return {"message": "Bine ai venit la Symptom Checker API!"}
