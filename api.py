import logging
import numpy as np
import torch
import japanese_clip as ja_clip
import io
import time
import uvicorn

from typing import List
from pydantic import BaseModel
from fastapi import Form, FastAPI, File, UploadFile, HTTPException
from PIL import Image
from typing import List, Dict

class CommonResponse(BaseModel):
    inference_time: float
    vector: List[float]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
device = "cpu"

# Load the CLIP model outside the route function (during server startup)
model, transform = ja_clip.load("rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip", device=device)
tokenizer = ja_clip.load_tokenizer()

logger.info("CLIP model loaded successfully.")

def img_to_vector(image: bytes) -> List[float]:
    try:
        # Convert binary data to PIL image
        image = Image.open(io.BytesIO(image))
        
        # Preprocess the image for the CLIP model
        image = transform(image).unsqueeze(0).to(device)

        # Forward the image through the model
        with torch.no_grad():
            image_features = model.encode_image(image)
        
        # Convert tensor to list
        result = image_features.cpu().numpy().tolist()

        return result

    except Exception as e:
        raise Exception("Failed to convert image to vector: " + str(e))

def text_to_vector(text: str) -> List[float]:
    try:
        # Preprocess the text for the CLIP model
        text = ja_clip.tokenize([text]).to(device)
        encodings = ja_clip.tokenize([text], max_seq_len=512, device=device, tokenizer=tokenizer)
        logging.info(f"Encodings from ja_clip.tokenize: {encodings}")

        # Forward the text through the model
        with torch.no_grad():
            text_features = model.get_text_features(**encodings)
        
        # Convert tensor to list
        result = text_features.cpu().numpy().tolist()[0]  # Assuming you want the list for the first (and only) text

        return result

    except Exception as e:
        raise Exception("Failed to convert text to vector: " + str(e))


@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/img2vec", response_model=CommonResponse, responses={
    200: {
        "description": "JSON object representing the inference time and the text vector",
        "content": {
            "application/json": {
                "schema": CommonResponse.schema()
            }
        }
    },
    400: {
        "description": "Bad request, the text is not present"
    }
})

async def convert_to_image_vector(image_data: UploadFile = Form(...)):
    logging.info(f"Received file: {image_data.filename}")
    try:
        # Read file contents
        contents = await image_data.read()
        start_time = time.time()
        # Get the vector representation of the image
        vector = img_to_vector(contents)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Prepare response as per the updated Swagger
        response = {"inference_time": inference_time, "vector": vector}

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.post("/txt2vec", response_model=CommonResponse, responses={
    200: {
        "description": "JSON object representing the inference time and the text vector",
        "content": {
            "application/json": {
                "schema": CommonResponse.schema()
            }
        }
    },
    400: {
        "description": "Bad request, the text is not present"
    }
})
async def convert_text_to_vector(text_data: str = Form(...)):
    try:
        start_time = time.time()

        # Preprocess the text for the CLIP model
        encodings = ja_clip.tokenize([text_data], max_seq_len=512, device=device, tokenizer=tokenizer)

        # Forward the text through the model
        with torch.no_grad():
            text_features = model.get_text_features(**encodings)

        # Calculate inference time
        inference_time = time.time() - start_time

        # Convert tensor to list and prepare response
        vector = text_features.cpu().numpy().tolist()[0]
        response = {"inference_time": inference_time, "vector": vector}

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
