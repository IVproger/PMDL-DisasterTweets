from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.main import make_prediction_ml, make_prediction_dl
from src.utils import init_hydra

app = FastAPI()

class ModelRequest(BaseModel):
    model_name: str
    version: int
    raw_text: str
    
class ModelInfo(BaseModel):
    model_name: str
    version: int
    type: str
class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    
def get_model_names() -> ModelsResponse:
    try:
        cfg = init_hydra("models_description.yaml")
        models = []
        for model in cfg['production']['models']:
            models.append(ModelInfo(model_name=model['name'], version=model['version'],type=model['type']))
        return ModelsResponse(models=models)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail="Error loading models")

@app.post("/predict_ml")
async def predict(model_request: ModelRequest):
    try:
        prediction = make_prediction_ml(model_request.model_name+'_v'+str(model_request.version)+'.pkl', model_request.raw_text)
        print("Prediction completed")
        print("Prediction:", prediction)
        return {"prediction": int(prediction)}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_dl")
async def predict(model_request: ModelRequest):
    try:
        prediction = make_prediction_dl(model_request.model_name, model_request.raw_text)
        print("Prediction completed")
        print("Prediction:", prediction)
        return {"prediction": int(prediction)}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models_list", response_model=ModelsResponse)
async def get_models():
    models = get_model_names()
    return models

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)