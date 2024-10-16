from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.main import make_prediction_ml
from src.utils import init_hydra
import os

app = FastAPI()

class ModelRequest(BaseModel):
    model_name: str
    version: str
    raw_text: str
    
class ModelInfo(BaseModel):
    model_name: str
    version: str
class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    
def get_model_names() -> ModelsResponse:
    try:
        cfg = init_hydra("models_description.yaml")
        print(cfg)
        models_dir = cfg.production.models_path
        model_files = os.listdir(models_dir)
        models = []
        for file in model_files:
            if file.endswith(".pkl") and "_" in file:
                name, version_with_ext = file.rsplit("_v", 1)
                version = version_with_ext.rsplit(".", 1)[0] 
                models.append(ModelInfo(model_name=name, version=version))
        return ModelsResponse(models=models)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail="Error loading models")

@app.post("/predict_ml")
async def predict(model_request: ModelRequest):
    try:
        prediction = make_prediction_ml(model_request.model_name, model_request.raw_text)
        print("Prediction completed")
        print("Prediction:", prediction)
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_dl")
async def predict(model_request: ModelRequest):
    pass

@app.get("/models_list", response_model=ModelsResponse)
async def get_models():
    models = get_model_names()
    return models

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)