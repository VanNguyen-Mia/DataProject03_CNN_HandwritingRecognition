import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

# ========== Functions ===========#
def center_image(image_array):
    # Find the non-zero pixels
    non_zero_pixels = np.where(image_array > 0)
    min_y, max_y = np.min(non_zero_pixels[0]), np.max(non_zero_pixels[0])
    min_x, max_x = np.min(non_zero_pixels[1]), np.max(non_zero_pixels[1])

    # Crop the image to the bounding box of the drawn object
    cropped = image_array[min_y:max_y+1, min_x:max_x+1]

    # Create a new image (28x28) and paste the cropped image in the center
    centered_image = np.zeros((28, 28))
    y_offset = (28 - cropped.shape[0]) // 2
    x_offset = (28 - cropped.shape[1]) // 2
    centered_image[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped

    return centered_image

# ========== Intro Text ==========
st.title('MNIST-Trained Handwritten Digit OCR')
st.write('GitHub: https://github.com/cwfrock')
st.write('Using a convolutional neural network (CNN) trained on the famous MNIST dataset, harness the power of AI to recognize your handwriting!')
st.subheader('Note About Data Privacy')
st.write("""Any upload, drawing, or camera input is hosted by Streamlit\'s third-party cloud providers 
         and is encrypted in transit via 256-bit encryption. No data will be visible or accessible by the owner of this app.
         See https://docs.streamlit.io/streamlit-community-cloud/get-started/trust-and-security for more information.""")
st.subheader("ML Model Background")
st.write("""
        The model used to calculate probabilities and predictions for handwriting input is a simple CNN trained on the standard MNIST dataset. Each
         image in MNIST is 28 x 28 pixels. 
         """)

# ========== Credits ==========
st.subheader("Credits")
st.markdown(" * `streamlit-drawable-canvas`: https://github.com/andfanilo/streamlit-drawable-canvas")
st.markdown(" * **Streamlit**: https://streamlit.io/")
st.markdown(" * *Machine Learning with PyTorch and Scikit-Learn* by Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili: https://sebastianraschka.com/books/#machine-learning-with-pytorch-and-scikit-learn")

# ========== User Input ==========
st.subheader('Handwriting Input')
input_type = st.radio("Choose how you would like to input a sample of your handwritten digit:",
                      ["Upload a .png!", "Draw your digit!"])

# ========== Case: PNG ==========
if input_type == "Upload a .png!":
    st.write('Upload a sample of a handwritten digit (0-9) in .png format.')
    uploaded_file = st.file_uploader(label = 'Upload your file in .png format!', type = 'png')
    if uploaded_file is not None:
        user_img = Image.open(uploaded_file)
        # Resize and convert to grayscale for Keras model
        convert_user_input_to_tensor = tf.keras.preprocessing.image.img_to_array(user_img.resize((28, 28)).convert('L')) / 255.0
        # Add batch dimension for model input
        input_tensor = np.expand_dims(convert_user_input_to_tensor, axis=0)
        st.image(user_img)
    else:
        st.warning("No sample has been found!")
        st.stop()

# ========== Case: Draw ==========
elif input_type == "Draw your digit!":
    st.subheader('Tip!')
    st.write('Change the stroke width and see if this changes the model\'s prediction.')
    stroke_width = st.slider("Stroke width: ", 1, 100, 30)
    canvas_result = st_canvas(background_color = "#ffffff", 
                              stroke_width = stroke_width, 
                              update_streamlit = True, 
                              height = 400, 
                              width = 400, 
                              display_toolbar = True)
    
    form = st.form(key='my_form')
    submitted_input = form.form_submit_button('Submit!')
        
    if submitted_input:
        user_img = Image.fromarray(canvas_result.image_data.astype(np.uint8))

        # Convert to grayscale (L mode) and resize
        user_img_resized = user_img.resize((28, 28)).convert('L')

        # Convert to NumPy array
        user_img_array = np.array(user_img_resized)

        # Center the image (if drawn digits are off-center)
        centered_image = center_image(user_img_array)

        # Convert the centered image to a tensor (for model input)
        convert_user_input_to_tensor = tf.keras.preprocessing.image.img_to_array(centered_image) / 255.0

        # Add batch dimension for model input
        input_tensor = np.expand_dims(convert_user_input_to_tensor, axis=0)

        st.image(user_img_resized)
    else:
        st.warning("No sample has been found!")
        st.stop()

# ========== Load Keras model and predict ==========
model = load_model("./saved_model/cnn_model.keras")  # Loading the Keras model

if input_type == "Upload a .png!":
    pred = model.predict(input_tensor)
elif input_type == "Draw your digit!":
    pred = model.predict(input_tensor)

# Get prediction
user_number_pred = np.argmax(pred, axis=1)[0]

# Display the prediction
st.subheader('Prediction')
st.write(f'The model predicts that you wrote the number {user_number_pred} with a probability of {pred[0][user_number_pred] * 100:.2f}%!')

# ========== Plotly Charts ==========
st.subheader('Softmax Probabilities')
probas = np.round(pred[0], 2)
df = pd.DataFrame({"digits": np.arange(10), "probability": probas})
fig = px.bar(df, x = "digits", y = "probability", color = "probability", text = "probability", color_continuous_scale = "thermal")
fig.update_traces(textposition = 'inside')
st.plotly_chart(fig)