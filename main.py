import io
import pickle
import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

with open('mnist_model.pk', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = PIL.Image.open(io.BytesIO(await file.read()))
    image = PIL.ImageOps.grayscale(image)
    image = image.resize((28, 28), PIL.Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
    image = np.array(image).reshape(1, -1) / 255.0  # Reshape to 2D array
    prediction = model.predict(image)
    return {"prediction": int(prediction[0])}
