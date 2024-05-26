import streamlit as st
import requests
from PIL import Image
import io

# Title of the application
st.title("Chessman Classification")

# About the project
st.subheader("About this Project")
st.write("This application allows you to classify images of chess pieces. Upload an image of a chess piece, and the model will predict whether it is a Bishop, King, Knight, Pawn, Queen, or Rook.")

# How to use the application
st.subheader("How to Use")
st.write("""
1. Click on the Browse files button to upload an image of a chess piece.
2. Once the image is uploaded, it will be displayed on the screen.
3. Click the Predict button to get the classification result.
4. The predicted class will be displayed below the image.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_bytes = image_bytes.getvalue()

    # Predict button
    if st.button('Predict'):
        files = {'file': image_bytes}
        try:
            with st.spinner("Predicting..."):
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                response.raise_for_status()
                result = response.json()
                st.write(f"Predicted Class: {result['predicted_class']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

# Instructions to run the Streamlit app
st.write("Run the following command in your terminal to start the Streamlit app:")
st.code("streamlit run src/app.py")
