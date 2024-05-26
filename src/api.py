from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io

print("api.py is being executed")  # Added  this line for debugging

app = FastAPI()
model = load_model('notebooks/best_model.keras')
class_names = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']  # Predicted class names

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Reads the file contents into memory
        contents = await file.read()
        
        # Convert the file contents to a PIL Image
        img = Image.open(io.BytesIO(contents))
        img = img.resize((150, 150))  # Resizes image to match the model's expected input size
        
        # Converts the image to a numpy array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Makes prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction[0])]
        return {"predicted_class": predicted_class}

    except Exception as e:
        # Logs the error and raise an HTTP exception
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
