import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_drawable_canvas import st_canvas
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_trained_model, preprocess_image, predict_digit

MODEL_PATH = "./saved_model/cnn_model.keras"


# ========== Streamlit UI ==========
st.title('MNIST Handwritten Digit Recognition')
st.write('GitHub: [View Project](https://github.com/cwfrock)')
st.write("A CNN model trained on the MNIST dataset recognizes your handwritten digits.")

st.subheader("Choose Input Method")
input_type = st.radio("Select an input method:", ["Upload a .png!", "Draw your digit!"])

model = load_trained_model(MODEL_PATH)

# ========== Case: PNG Upload ==========
if input_type == "Upload a .png!":
    st.write('Upload a sample of a handwritten digit (0-9) in .png format.')
    uploaded_file = st.file_uploader("Upload a handwritten digit image (.png)", type=["png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and Predict
        processed_image = preprocess_image(image)
        digit, prediction, confidence = predict_digit(model, processed_image)

        # Display Results
        st.subheader("Prediction")
        st.write(f"Predicted Number: **{digit}** with **{confidence:.2f}%** confidence.")
        
        # Plot Softmax Probabilities
        prob_df = pd.DataFrame({"Digits": np.arange(10), "Probability": np.round(prediction[0], 2)})
        fig = px.bar(prob_df, x="Digits", y="Probability", color="Probability", text="Probability", color_continuous_scale="thermal")
        fig.update_traces(textposition="inside")
        st.plotly_chart(fig)
    else:
        st.warning("Please upload an image.")

# ========== Case: Draw Digit ==========
elif input_type == "Draw your digit!":
    st.subheader('Tip!')
    st.write('Change the stroke width and see if this changes the model\'s prediction.')
    st.write("Draw your digit in the box below:")
    stroke_width = st.slider("Stroke width:", 1, 50, 25)
    
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=400,
        width=400,
        display_toolbar=True
    )

    form = st.form(key='my_form')
    
    if st.button("Submit Drawing"):
        if canvas_result.image_data is not None:
            user_image = Image.fromarray(canvas_result.image_data.astype(np.uint8))
            processed_image = preprocess_image(user_image)

            # Predict
            digit, prediction, confidence = predict_digit(model, processed_image)

            # Display Results
            st.subheader("Prediction")
            st.write(f"Predicted Number: **{digit}** with **{confidence:.2f}%** confidence.")
            
            # Plot Softmax Probabilities
            prob_df = pd.DataFrame({"Digits": np.arange(10), "Probability": prediction[0]})
            fig = px.bar(prob_df, x="Digits", y="Probability", color="Probability", text="Probability", color_continuous_scale="thermal")
            fig.update_traces(textposition="inside")
            st.plotly_chart(fig)
        else:
            st.warning("No drawing detected. Please draw a digit.")

# ========== Credits ==========
st.subheader("Credits")
st.markdown(" * [Streamlit](https://streamlit.io/)")
st.markdown(" * [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)")
st.markdown(" * *Machine Learning with PyTorch and Scikit-Learn* by Sebastian Raschka")
