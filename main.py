import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

st.title('Handwritten Digit Recognition')

input_model = open('lrc_mnist.pkl', 'rb')
model = pkl.load(input_model)

st.header('Upload handwritten digit image')
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Test image', use_column_width=True)

    if st.button('Predict'):
        # Resize image and convert to a flat array
        image = image.resize((8, 8))  # Adjust the size accordingly
        feature_vector = np.array(image).flatten()

        # Normalize pixel values to be between 0 and 1 (if needed)
        feature_vector = feature_vector / 255.0

        # Make prediction
        label = str((model.predict([feature_vector]))[0])

        st.header('Result')
        st.text(f'Predicted digit: {label}')
else:
    st.info('Please upload an image.')
