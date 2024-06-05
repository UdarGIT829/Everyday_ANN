from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import os
from Network import train_model, run_saved_model
from easydict import EasyDict
from tempfile import NamedTemporaryFile
import uvicorn
import uuid

# Initialize FastAPI app
app = FastAPI()

DEFAULT_trial_details = EasyDict({
    "hidden_layer_sizes": {"low": 4, "high": 128},
    "num_epochs": {"low": 50, "high": 1000},
    "learning_rate": {"low": 1e-5, "high": 1e-1},
    "batch_size": {"low": 16, "high": 128},
    "activation": ['ReLU', 'Tanh', 'LeakyReLU', 'ELU', 'GELU', 'Swish']
})

class TrainModelRequest(BaseModel):
    trial_details: Dict[str, Any] = {}
    n_trials: int = 5

class PredictionRequest(BaseModel):
    input_data: Dict[str, Any]
    model_id: str

@app.post("/train")
async def train_model_endpoint(
    trial_details: str = Form(...),
    n_trials: int = Form(...),
    data_file: UploadFile = File(...),
    return_file: bool = Query(False)
):
    """
    Train a model with the provided data file and trial details.
    
    Args:
        trial_details (str): JSON string containing trial details.
        n_trials (int): Number of trials for the training.
        data_file (UploadFile): Uploaded data file for training the model.
        return_file (bool): Query parameter to determine if the trained model file should be returned.
        
    Returns:
        JSON response with a success message and model path, or the model file if `return_file` is True.
    """
    data_filename = os.path.splitext(data_file.filename)[0]
    uid = str(uuid.uuid4())
    model_id = f"{data_filename}_{uid}"
    model_path = f"{model_id}.pkl"
    trial_details_dict = EasyDict(eval(trial_details))  # Convert the JSON string to a dictionary

    # Save the uploaded data file to a temporary location
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await data_file.read())
        data_path = tmp.name

    try:
        train_model(model_path, data_path, trial_details_dict, n_trials)
    finally:
        os.remove(data_path)  # Clean up the temporary data file

    if return_file:
        return FileResponse(model_path, filename=f"{model_id}.pkl", media_type='application/octet-stream')

    return {"message": "Model trained and saved successfully.", "model_id": model_id}

@app.post("/predict")
async def predict_endpoint(
    request: PredictionRequest,
    data_file: UploadFile = File(None)
):
    """
    Run predictions using the provided model ID and input data file.
    
    Args:
        request (PredictionRequest): Request body containing input data for predictions and model ID.
        data_file (UploadFile, optional): Uploaded data file for predictions.
        
    Returns:
        JSON response with predictions.
        
    Raises:
        HTTPException: If the model or data file is not found.
    """
    model_path = f"{request.model_id}.pkl"

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found. Train the model first or provide a valid model ID.")

    if data_file:
        # Save the uploaded data file to a temporary location
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await data_file.read())
            data_path = tmp.name
    else:
        raise HTTPException(status_code=400, detail="Data file must be provided.")

    try:
        prediction = run_saved_model(model_path, request.input_data, data_path)
    finally:
        os.remove(data_path)  # Clean up the temporary data file

    return {"predictions": prediction}

@app.post("/predict_from_file")
async def predict_from_file(
    prediction_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    model_id: str = Form(...)
):
    """
    Run predictions using the uploaded prediction data file, input data file, and model ID.
    
    Args:
        prediction_file (UploadFile): Uploaded prediction data file.
        data_file (UploadFile): Uploaded input data file.
        model_id (str): ID of the trained model to use for predictions.
        
    Returns:
        JSON response with predictions.
        
    Raises:
        HTTPException: If the model is not found.
    """
    model_path = f"{model_id}.pkl"

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found. Train the model first or provide a valid model ID.")

    # Save the uploaded prediction data file to a temporary location
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await prediction_file.read())
        prediction_path = tmp.name

    # Save the uploaded input data file to a temporary location
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await data_file.read())
        data_path = tmp.name

    try:
        prediction = run_saved_model(model_path, prediction_path, data_path)
    finally:
        os.remove(prediction_path)  # Clean up the temporary prediction data file
        os.remove(data_path)  # Clean up the temporary input data file

    return {"predictions": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)