from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
import base64
import logging
import sys
import logging

root = logging.getLogger()
root.setLevel(logging.DEBUG)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
root.addHandler(stdout_handler)

logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG


app=Flask(__name__)
CORS(app)  # Enable CORS for all routes

model=tf.keras.models.load_model("C:/Users/Nihala/OneDrive/Desktop/potato-disease/training/mymodel.h5")
model_class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']  # Define the class names

@app.route('/api/predict',methods=['POST'])
def predict():
    try:
        # Process the image and make the prediction
        # Get the uploaded image from the request
        image_file = request.files['image']

        # Save the uploaded image to a temporary location
        upload_path = "C:/Users/Nihala/OneDrive/Desktop/potato-disease/training"  # Specify the directory to save the file
        image_file.save(os.path.join(upload_path, 'uploaded_image.jpg'))
        image_path = os.path.join(upload_path, 'uploaded_image.jpg')

        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        logging.debug("Image loaded successfully.")
        logging.info("Original image dimensions: %s", img.size)
        img = img.resize((256, 256))
        logging.info("Resized image dimensions: %s", img.size)
        img_array = img_to_array(img)
        logging.info("Image array (before normalization):\n%s", img_array)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        logging.info("Image array (after normalization):\n%s", img_array)

        # Perform prediction using the loaded model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Encode the image as base64
        with open(image_path, "rb") as file:
            image_data = file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Return the predicted class and image as JSON response
        disease_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        logging.info("Web Page Code - disease_labels:")
        logging.info(disease_labels)
        response = {
            'disease': disease_labels[predicted_class],
            'image': base64_image
        }

        # Verify if `disease_labels` has the same length as `model_class_names`
        if len(disease_labels) != len(model_class_names):
            logging.error("Error: The number of labels does not match the number of output classes.")
        else:
            # Compare the elements of `disease_labels` and `model_class_names` pairwise
            for label, class_name in zip(disease_labels, model_class_names):
                if label != class_name:
                    logging.error("Error: The labels and class_names do not match.")
                    break
            else:
                logging.info("The labels and class_names match.")

        return jsonify(response)
    except Exception as e:
        logging.error("Error occurred while making the prediction: %s", str(e))
        return jsonify({'error': 'Error occurred while making the prediction.'})


if __name__ == '__main__':
    model.summary()
    app.run()
