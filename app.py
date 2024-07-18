from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from Network import train_model, run_saved_model, hash_file_start
from easydict import EasyDict
from tempfile import NamedTemporaryFile
import uvicorn
import uuid
from starlette.requests import Request

from DataImporter import DataImporter
import pickle

# Initialize FastAPI app
app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

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
    user_model_id: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/load_model")
async def load_model(model_file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await model_file.read())
        model_path = tmp.name
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        training_metadata = model.training_metadata
        column_headers = model.column_headers
    finally:
        os.remove(model_path)
    
    return JSONResponse(content={"training_metadata": training_metadata, "column_headers": column_headers})

@app.post("/load_data_file")
async def load_data_file(data_file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await data_file.read())
        data_path = tmp.name

    try:
        data_importer = DataImporter(data_path)
        column_headers = data_importer.column_names
    finally:
        os.remove(data_path)
    
    return JSONResponse(content={"column_headers": column_headers})

@app.post("/validate_model_data_file")
async def validate_model_data_file(model_file: UploadFile = File(...), data_file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False) as tmp_model:
        tmp_model.write(await model_file.read())
        model_path = tmp_model.name

    with NamedTemporaryFile(delete=False) as tmp_data:
        tmp_data.write(await data_file.read())
        data_path = tmp_data.name

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        data_hash = hash_file_start(data_path)
        if data_hash == model.training_metadata.get("data_hash"):
            message = "Validation successful: Data file matches the model's training data."
        else:
            message = "Validation failed: Data file does not match the model's training data."
    finally:
        os.remove(model_path)
        os.remove(data_path)

    return JSONResponse(content={"message": message})

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
    user_model_id = f"{data_filename}_{uid}"
    model_path = f"{user_model_id}.pkl"
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
        return FileResponse(model_path, filename=f"{user_model_id}.pkl", media_type='application/octet-stream')

    return {"message": "Model trained and saved successfully.", "user_model_id": user_model_id}

@app.post("/predict")
async def predict_endpoint(
    data_file: UploadFile = File(...),
    user_model_id: Optional[str] = Form(None),
    user_model_file: Optional[UploadFile] = File(None),
    input_data: Optional[str] = Form(None),
    prediction_file: Optional[UploadFile] = File(None)
):
    """
    Run predictions using the provided model ID or model file and input data or prediction file.
    
    Args:
        data_file (UploadFile): Uploaded data file for predictions.
        user_model_id (str, optional): ID of the trained model to use for predictions.
        user_model_file (UploadFile, optional): Uploaded model file for predictions.
        input_data (str, optional): JSON string containing input data for predictions.
        prediction_file (UploadFile, optional): Uploaded file containing prediction input data.
        
    Returns:
        JSON response with predictions.
        
    Raises:
        HTTPException: If the model or data file is not found.
    """
    if user_model_id:
        model_path = f"{user_model_id}.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found. Train the model first or provide a valid model ID.")
    elif user_model_file:
        # Save the uploaded model file to a temporary location
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await user_model_file.read())
            model_path = tmp.name
    else:
        raise HTTPException(status_code=400, detail="Either user_model_id or user_model_file must be provided.")

    # Save the uploaded data file to a temporary location
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await data_file.read())
        data_path = tmp.name

    if prediction_file:
        # Save the uploaded prediction data file to a temporary location
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await prediction_file.read())
            prediction_path = tmp.name

        input_data_dict=prediction_path
    elif input_data:
        input_data_dict = eval(input_data)
    else:
        raise HTTPException(status_code=400, detail="Either input_data or prediction_file must be provided.")

    try:
        prediction = run_saved_model(model_path, input_data_dict, data_path)
    finally:
        os.remove(data_path)  # Clean up the temporary data file
        if user_model_file:
            os.remove(model_path)  # Clean up the temporary model file


    return {"predictions": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8110)