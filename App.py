from flask import Flask, request, jsonify,make_response
import requests
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
import base64
import logging
import sys
import uuid
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt

app=Flask(__name__)
CORS(app)  # Enable CORS for all routes

model=tf.keras.models.load_model("C:/Users/Nihala/OneDrive/Desktop/potato-disease/training/mymodelnew.h5")

model_class_names = ['Others','Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


@app.route('/api/predict',methods=['POST'])
def predict():
    
    # Process the image and make the prediction
    # Get the uploaded image from the request
    image_file = request.files['image']
    

    # Save the uploaded image to a temporary location
    upload_path = "C:/Users/Nihala/OneDrive/Desktop/potato-disease/training"  # Specify the directory to save the file
    unique_identifier = str(uuid.uuid4())
    image_filename = f"uploaded_image_{unique_identifier}.jpg"
    image_file.save(os.path.join(upload_path, image_filename))
    image_path = os.path.join(upload_path, image_filename)

    # Load and preprocess the image
    img = Image.open(image_path)

    resized_img = img.resize((256,256))
   
    img_array = img_to_array(resized_img)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions=model.predict(img_batch)
    
    predicted_class=model_class_names[np.argmax(predictions[0])]
    confidence=np.max(predictions[0])

    # Encode the image as base64
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')


    return{
        "class":predicted_class,
        "confidence":float(confidence),
        "image":base64_image
        }
if __name__=="__main__":
    app.run()
     

