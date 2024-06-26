from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load your UNet model
model = tf.keras.models.load_model('unet_model.h5')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    data = request.get_json()
    image_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Preprocess the image as required by your UNet model
    image = image.resize((128, 128))  # Adjust size as necessary
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Get the prediction from your UNet model
    prediction = model.predict(image_array)
    predicted_color = analyze_prediction(prediction)

    return jsonify({'color': predicted_color})

def analyze_prediction(prediction):
    # Implement your logic to analyze the prediction and determine the color
    # For example, you could use a color threshold or average color method
    color = 'some_color'  # Placeholder
    return color

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
