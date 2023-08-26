import streamlit as st
from PIL import Image
# Importing relevant libraris
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

from tensorflow.keras.preprocessing.image import load_img, img_to_array


def get_category(value):
    if(value == 0):
        return 'Category 1 Tumor'
    elif(value == 1):
        return 'Category 2 Tumor'
    elif(value == 2):
        return 'Category 3 Tumor'
    else:
        return 'No Tumor'

# Title
st.title("Brain Tumor Detection")

loaded_model = tf.keras.models.load_model("datathon_bytestormsquad_.h5")

# Image upload button
uploaded_file = st.file_uploader("Choose an image...", type="jpg")


if uploaded_file is not None:
    # Read and preprocess the uploaded image
    image = load_img(uploaded_file, target_size=(150, 150))  # Resize to match your model's input size
    image_array = img_to_array(image)
    

    # Expand dimensions to create a batch of one image
    image_batch = np.expand_dims(image_array, axis=0)

    # Make a prediction
    prediction = loaded_model.predict(image_batch)

    value = np.argmax(prediction)


st.write("Upload the MRI scan Image file here")

# Display the uploaded image if available
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Prediction button
if st.button('Predict'):
    st.write('The patient is having ')
    st.subheader(get_category(value))
