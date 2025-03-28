import matplotlib.pyplot as plt
import tensorflow as tf


def show_and_save_image(image, title, filename):
    """
    Display and save the image at the current preprocessing step.
    """
    # Add channel dimension if missing
    if len(image.shape) == 2:  # If the image is grayscale without channel dimension
        image = tf.expand_dims(image, axis=-1)  # Add channel dimension
    
    # Display the image
    plt.imshow(tf.squeeze(image), cmap='gray')  # Squeeze to remove batch dimensions for display
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    # Save the image
    tf.keras.utils.save_img(filename, image, scale=True)
    print(f"Saved: {filename}")


def preprocess_image(image_path, save_dir="./preprocessed_images/"):
    # Ensure the save directory exists
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Step 1: Read the file
    img = tf.io.read_file(image_path)
    
    # Step 2: Decode PNG (Grayscale)
    img = tf.io.decode_png(img, channels=1)  # Ensure it's grayscale
    show_and_save_image(img, "After Decoding (Grayscale)", os.path.join(save_dir, "step_1_decoded.png"))
    
    # Step 3: Convert to float32
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32
    show_and_save_image(img, "After Converting to Float32", os.path.join(save_dir, "step_2_float32.png"))
    
    # Step 4: Resize the image
    img_height, img_width = 80, 200  # Example dimensions
    img = tf.image.resize(img, [img_height, img_width])  # Resize image
    show_and_save_image(img, "After Resizing", os.path.join(save_dir, "step_3_resized.png"))
    
    # Step 5: Transpose the image
    img = tf.transpose(img, perm=[1, 0, 2])  # Transpose to match the model input format
    show_and_save_image(img, "After Transposing", os.path.join(save_dir, "step_4_transposed.png"))
    
    # Step 6: Add batch dimension
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    show_and_save_image(tf.squeeze(img), "After Adding Batch Dimension", os.path.join(save_dir, "step_5_batch_dimension.png"))
    
    return img

# Example usage
image_path = "/Users/vineeshreddy/Downloads/DATASETS/set-1/1mhjl9.png" 
preprocess_image(image_path)# Replace with your image path


