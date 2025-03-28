# import os

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# # Constants and model loading (same as your existing code)
# img_width = 200
# img_height = 80
# max_length = 6
# char_img = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# char_to_num = layers.StringLookup(vocabulary=list(char_img), mask_token=None)
# num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# # Custom CTC layer (same as your existing code)
# class LayerCTC(keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(LayerCTC, self).__init__(**kwargs)
#         self.loss_fn = keras.backend.ctc_batch_cost

#     def call(self, y_true, y_pred):
#         batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
#         input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
#         label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

#         input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#         label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

#         loss = self.loss_fn(y_true, y_pred, input_length, label_length)
#         self.add_loss(loss)

#         return y_pred

#     @classmethod
#     def from_config(cls, config):
#         config.pop('trainable', None)
#         return cls(**config)

# # Load the trained model (same as your existing code)
# model_path = "/Users/vineeshreddy/Desktop/CAPTCHA-FINAL/MODELS/DATASET.h5"
# model = keras.models.load_model(model_path, custom_objects={'CTCLayer': LayerCTC})

# # Create prediction model (same as your existing code)
# input_img = model.input[0]
# output_pred = model.get_layer(name="dense2").output
# prediction_model = keras.models.Model(inputs=input_img, outputs=output_pred)

# # Decode the predictions (same as your existing code)
# def decode_batch_predictions(pred):
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]
#     results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]

#     output_text = []
#     for res in results:
#         res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
#         output_text.append(res)
#     return output_text

# # Preprocess the image (same as your existing code)
# def preprocess_image(image_path):
#     img = tf.io.read_file(image_path)
#     img = tf.io.decode_png(img, channels=1)  # Ensure it's grayscale
#     img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32
#     img = tf.image.resize(img, [img_height, img_width])  # Resize image
#     img = tf.transpose(img, perm=[1, 0, 2])  # Transpose to match the model input format
#     img = tf.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# # Function to extract the ground truth text (assuming it's in the file name)
# def get_ground_truth(image_path):
#     # Assume the file name is the ground truth text, e.g., '123456.png'
#     return os.path.basename(image_path).split('.')[0]
# def predict_single_image(image_path):
#     # Preprocess the image
#     img = preprocess_image(image_path)

#     # Get the prediction
#     preds = prediction_model.predict(img)
    
#     # Decode the predictions to text
#     pred_texts = decode_batch_predictions(preds)

#     return pred_texts[0]  # Return the predicted text for the single image
# # Test Accuracy Calculation
# def calculate_accuracy(input_folder):
#     correct_predictions = 0
#     total_predictions = 0

#     # Loop through all the image files in the folder
#     for root, dirs, files in os.walk(input_folder):
#         for file in files:
#             if file.endswith(".png"):  # Only process PNG files (or adjust if using other formats)
#                 image_path = os.path.join(root, file)
                
#                 # Get ground truth text
#                 ground_truth = get_ground_truth(image_path)
                
#                 # Preprocess image and get prediction
#                 predicted_text = predict_single_image(image_path)

#                 # Compare predicted text with ground truth
#                 if predicted_text == ground_truth:
#                     correct_predictions += 1
#                 total_predictions += 1

#     accuracy = correct_predictions / total_predictions
#     print(f"Accuracy: {accuracy * 100:.2f}%")

# # Example: Provide the path to your input folder
# input_folder = "/Users/vineeshreddy/Downloads/voter_id_captachs_png"  # Replace with your actual input folder path
# calculate_accuracy(input_folder)



# import os

# import numpy as np
# import tensorflow as tf
# from sklearn.metrics import f1_score, precision_score, recall_score
# from tensorflow import keras
# from tensorflow.keras import layers

# # Constants and model loading
# img_width = 200
# img_height = 80
# max_length = 6
# char_img = [
#     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
#     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
#     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#     'v', 'w', 'x', 'y', 'z'
# ]
# char_to_num = layers.StringLookup(vocabulary=list(char_img), mask_token=None)
# num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# # Custom CTC layer
# class LayerCTC(keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(LayerCTC, self).__init__(**kwargs)
#         self.loss_fn = keras.backend.ctc_batch_cost

#     def call(self, y_true, y_pred):
#         batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
#         input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
#         label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

#         input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#         label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

#         loss = self.loss_fn(y_true, y_pred, input_length, label_length)
#         self.add_loss(loss)

#         return y_pred

#     @classmethod
#     def from_config(cls, config):
#         config.pop('trainable', None)
#         return cls(**config)

# # Load the trained model
# model_path = "/Users/vineeshreddy/Desktop/CAPTCHA-FINAL/MODELS/ALL_IN_ONE.h5"
# model = keras.models.load_model(model_path, custom_objects={'CTCLayer': LayerCTC})

# # Create prediction model
# input_img = model.input[0]
# output_pred = model.get_layer(name="dense2").output
# prediction_model = keras.models.Model(inputs=input_img, outputs=output_pred)

# # Decode the predictions
# def decode_batch_predictions(pred):
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]
#     results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]

#     output_text = []
#     for res in results:
#         res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
#         output_text.append(res)
#     return output_text

# # Preprocess the image
# def preprocess_image(image_path):
#     img = tf.io.read_file(image_path)
#     img = tf.io.decode_png(img, channels=1)  # Ensure it's grayscale
#     img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32
#     img = tf.image.resize(img, [img_height, img_width])  # Resize image
#     img = tf.transpose(img, perm=[1, 0, 2])  # Transpose to match the model input format
#     img = tf.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# # Extract the ground truth text
# def get_ground_truth(image_path):
#     return os.path.basename(image_path).split('.')[0]

# # Predict single image
# def predict_single_image(image_path):
#     img = preprocess_image(image_path)
#     preds = prediction_model.predict(img)
#     pred_texts = decode_batch_predictions(preds)
#     return pred_texts[0]

# # Calculate metrics
# from sklearn.metrics import (accuracy_score, f1_score, precision_score,
#                              recall_score)


# # Calculate metrics
# def calculate_metrics(input_folder):
#     all_ground_truths = []
#     all_predictions = []

#     for root, dirs, files in os.walk(input_folder):
#         for file in files:
#             if file.endswith(".png"):  # Only process PNG files
#                 image_path = os.path.join(root, file)

#                 ground_truth = get_ground_truth(image_path)
#                 predicted_text = predict_single_image(image_path)

#                 all_ground_truths.append(ground_truth)
#                 all_predictions.append(predicted_text)

#     # Convert to lists of same length by padding/truncating predicted_text
#     max_len = max_length
#     y_true = [list(gt.ljust(max_len)[:max_len]) for gt in all_ground_truths]
#     y_pred = [list(pred.ljust(max_len)[:max_len]) for pred in all_predictions]

#     # Flatten for character-level metrics
#     y_true_flat = [char for seq in y_true for char in seq]
#     y_pred_flat = [char for seq in y_pred for char in seq]

#     # Calculate character-level metrics
#     precision = precision_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
#     recall = recall_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
#     f1 = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)

#     # Calculate CAPTCHA-level accuracy
#     correct_predictions = sum(1 for gt, pred in zip(all_ground_truths, all_predictions) if gt == pred)
#     total_predictions = len(all_ground_truths)
#     accuracy = correct_predictions / total_predictions

#     print(f"Accuracy (CAPTCHA-level): {accuracy * 100:.2f}%")
#     print(f"Precision (Character-level): {precision * 100:.2f}%")
#     print(f"Recall (Character-level): {recall * 100:.2f}%")
#     print(f"F1 Score (Character-level): {f1 * 100:.2f}%")


# # Example usage
# input_folder = "/Users/vineeshreddy/Downloads/voter_id_captachs_png"  # Replace with your actual input folder path
# calculate_metrics(input_folder)

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers

# Constants and model loading
img_width = 200
img_height = 80
max_length = 6
char_img = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
    'v', 'w', 'x', 'y', 'z'
]
char_to_num = layers.StringLookup(vocabulary=list(char_img), mask_token=None)
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

        return y_pred

    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)
        return cls(**config)

# Load the trained model
model_path = "/Users/vineeshreddy/Desktop/CAPTCHA-FINAL/MODELS/DATASET.h5"
model = keras.models.load_model(model_path, custom_objects={'CTCLayer': LayerCTC})

# Create prediction model
input_img = model.input[0]
output_pred = model.get_layer(name="dense2").output
prediction_model = keras.models.Model(inputs=input_img, outputs=output_pred)

# Decode the predictions
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

# Preprocess the image
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=1)  # Ensure it's grayscale
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32
    img = tf.image.resize(img, [img_height, img_width])  # Resize image
    img = tf.transpose(img, perm=[1, 0, 2])  # Transpose to match the model input format
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Extract the ground truth text
def get_ground_truth(image_path):
    return os.path.basename(image_path).split('.')[0]

# Predict single image
def predict_single_image(image_path):
    img = preprocess_image(image_path)
    preds = prediction_model.predict(img)
    pred_texts = decode_batch_predictions(preds)
    return pred_texts[0]

# Calculate metrics and visualize
def calculate_metrics(input_folder):
    all_ground_truths = []
    all_predictions = []

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".png"):  # Only process PNG files
                image_path = os.path.join(root, file)

                ground_truth = get_ground_truth(image_path)
                predicted_text = predict_single_image(image_path)

                all_ground_truths.append(ground_truth)
                all_predictions.append(predicted_text)

    # Convert to lists of same length by padding/truncating predicted_text
    max_len = max_length
    y_true = [list(gt.ljust(max_len)[:max_len]) for gt in all_ground_truths]
    y_pred = [list(pred.ljust(max_len)[:max_len]) for pred in all_predictions]

    # Flatten for character-level metrics
    y_true_flat = [char for seq in y_true for char in seq]
    y_pred_flat = [char for seq in y_pred for char in seq]

    # Calculate character-level metrics
    precision = precision_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)

    # Calculate CAPTCHA-level accuracy
    correct_predictions = sum(1 for gt, pred in zip(all_ground_truths, all_predictions) if gt == pred)
    total_predictions = len(all_ground_truths)
    accuracy = correct_predictions / total_predictions

    # Print metrics
    print(f"Accuracy (CAPTCHA-level): {accuracy * 100:.2f}%")
    print(f"Precision (Character-level): {precision * 100:.2f}%")
    print(f"Recall (Character-level): {recall * 100:.2f}%")
    print(f"F1 Score (Character-level): {f1 * 100:.2f}%")

    # Plot metrics
    metrics = {'Accuracy': accuracy * 100, 'Precision': precision * 100, 'Recall': recall * 100, 'F1 Score': f1 * 100}
    
    # Create a bar plot
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Percentage (%)')
    plt.show()

# Example usage
input_folder = "/Users/vineeshreddy/Downloads/voter_id_captachs_png"  # Replace with your actual input folder path
calculate_metrics(input_folder)
