import io
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

# Define constants
img_width = 200
img_height = 80
max_length = 8  # Maximum length of the CAPTCHA text

# Character set (adjust it based on your use case)


char_img=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_num = layers.StringLookup(vocabulary=list(char_img), mask_token=None)

# Convert integers to characters
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# Custom CTC layer
class LayerCTC(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LayerCTC, self).__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # Return the predictions (used during inference)
        return y_pred

    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)
        return cls(**config)

# Load the trained model with custom CTC layer
model_path = "../MODELS/DATASET.h5"  # Change this to your model path
model = keras.models.load_model(model_path, custom_objects={'CTCLayer': LayerCTC})

# Create a prediction model that gives output from the final dense layer
input_img = model.input[0]  # Accessing the input layer correctly
output_pred = model.get_layer(name="dense2").output  # Accessing the dense layer output
prediction_model = keras.models.Model(inputs=input_img, outputs=output_pred)

# Decode the predicted output
# def decode_batch_predictions(pred):
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]  # Assuming the sequence length is the width of the image
#     results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    
#     output_text = []
#     for res in results:
#         # Decode the numerical prediction to text using the num_to_char mapping
#         res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
#         output_text.append(res)
#     return output_text
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]  # Sequence length of predictions
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    
    output_text = []
    for res in results:
        # Decode numerical prediction to text using num_to_char mapping
        decoded_text = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        # Remove [UNK] tokens explicitly
        cleaned_text = decoded_text.replace('[UNK]', '').strip()
        output_text.append(cleaned_text)
    return output_text

# Preprocess the input image
def preprocess_image(image):
    img = tf.io.decode_png(image, channels=1)  # Ensure it's grayscale
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32
    img = tf.image.resize(img, [img_height, img_width])  # Resize image
    img = tf.transpose(img, perm=[1, 0, 2])  # Transpose to match the model input format
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Initialize Flask app
app = Flask(__name__)

# Prediction function
def predict_single_image(image):
    # Preprocess the image
    img = preprocess_image(image)

    # Get the prediction
    preds = prediction_model.predict(img)
    
    # Decode the predictions to text
    pred_texts = decode_batch_predictions(preds)

    return pred_texts[0]  # Return the predicted text for the single image

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image is in the request
    if 'file' not in request.files:
        print("no-file")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        print("empty file")
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image from the request and make prediction
        img = file.read()
        predicted_text = predict_single_image(img)
        print(predicted_text)
        return jsonify({'predicted_text': predicted_text})
    except Exception as e:
        print("errorr")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
