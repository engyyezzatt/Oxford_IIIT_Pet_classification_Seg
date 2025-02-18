import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

# Preprocess the input image for both models
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img


# Run classification model
def run_classification_model(model, img_array):
    pred = model.predict(img_array)  
    predicted_class = np.argmax(pred, axis=1)
    return predicted_class, pred


# Run segmentation model
def run_segmentation_model(model, img_array):
    pred = model.predict(img_array)  
    predicted_mask = (pred > 0.5).astype(np.uint8)  # Convert to binary (0 or 1)
    return predicted_mask * 255  # Scale to 0 (black) and 255 (white)


# Save predictions (Segmentation and Classification)
def save_predictions(img, predicted_class, predicted_mask, class_labels, output_dir="output"):
    # Save Classification prediction
    class_label = class_labels[predicted_class[0]]  # Assuming it's a single image
    with open(f"{output_dir}/classification_result.txt", "w") as f:
        f.write(f"Predicted Class: {class_label}\n")

    # Save Segmentation prediction
    plt.imsave(f"{output_dir}/segmentation_result.png", predicted_mask[0, :, :, 0], cmap="gray", vmin=0, vmax=255)

    # Save original image
    img.save(f"{output_dir}/original_image.jpg")

# Display results
def display_results(img, predicted_class, predicted_mask, class_labels):
    plt.figure(figsize=(15, 5))

    # Display original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Display classification result
    class_label = class_labels[predicted_class[0]]
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.title(f"Predicted Class: {class_label}")
    plt.axis('off')

    # Display segmentation result
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask[0, :, :, 0], cmap="gray", vmin=0, vmax=255)
    plt.title("Predicted Segmentation Mask")
    plt.axis('off')
    plt.show()

# Main function to run both models on the input image
def run_models_on_image(img_path, classification_model, segmentation_model, class_labels, output_dir="output_predictions"):
    img_array_classification, img = preprocess_image(img_path, target_size=(224, 224))
    img_array_segmentation, _ = preprocess_image(img_path, target_size=(256, 256))  # Resize for segmentation model

    with ThreadPoolExecutor() as executor:
        class_future = executor.submit(run_classification_model, classification_model, img_array_classification)
        seg_future = executor.submit(run_segmentation_model, segmentation_model, img_array_segmentation)

        predicted_class, class_pred = class_future.result()
        predicted_mask = seg_future.result()

    display_results(img, predicted_class, predicted_mask, class_labels)
    save_predictions(img, predicted_class, predicted_mask, class_labels, output_dir)

    # Display results
    display_results(img, predicted_class, predicted_mask, class_labels)

    # Save predictions
    save_predictions(img, predicted_class, predicted_mask, class_labels, output_dir)

# Example usage:
if __name__ == "__main__":
    img_path = "data\classification_data\\test\yorkshire_terrier\yorkshire_terrier_28.jpg"
    
    # Load your models (Make sure to load your pre-trained models)
    classification_model = tf.keras.models.load_model('models\\classifier_final.h5')
    segmentation_model = tf.keras.models.load_model('models\UNet_Segmentor.h5')  # Replace with your model's path

    # Class labels 
    class_labels = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau',
                     'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog',
                       'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
                         'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond',
                           'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
                             'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

    # Run the models and display/save predictions
    run_models_on_image(img_path, classification_model, segmentation_model, class_labels)
