import os
import joblib
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, status 
from pydantic import BaseModel, Field

# URL where the model file is located on GitHub
github_url = "https://github.com/Chambeline-Nkah/Maths_ML_ALU/raw/main/regression.pkl"

# Download the model file
response = requests.get(github_url)
if response.status_code != 200:
    raise HTTPException(status_code=500, detail="Failed to download the model file")

# Define the directory path where the model file will be saved
directory_path = "./models"
os.makedirs(directory_path, exist_ok=True)

# Save the model file to the directory
model_path = os.path.join(directory_path, 'regression.pkl')
with open(model_path, 'wb') as f:
    f.write(response.content)

# Load the trained model
model = joblib.load(model_path)

app = FastAPI()

class PriceRequest(BaseModel):
    TV: int = Field(gt=0, lt=500)

@app.get("/greet")
async def get_greet():
    return {"Message": "Hello"}

@app.get("/", status_code=status.HTTP_200_OK)
async def get_hello():
    return {"hello": "world"}
     
@app.post('/predict', status_code=status.HTTP_200_OK)
async def make_prediction(price_request: PriceRequest):
    try:
        single_row = [[price_request.TV]]
        predicted_price = model.predict(single_row)
        return {"predicted_price": predicted_price[0][0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong.")
