import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

st.title('Handwritten Digit Recognition')

input = open('lrc_mnist.pkl', 'rb')
model = pkl.load(input)

st.header('Upload handwritten digit image')
uploaded_file = st.file_uploader("Choose an image file", type=(['png', 'jpg', 'jpeg']))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Test image')

    if st.button('Predict'):
        # Chuyển đổi ảnh thành ảnh đen trắng và resize
        image = image.convert('L').resize((8, 8))
        # Chuyển đổi ảnh thành mảng numpy
        feature_vector = np.array(image).reshape(1, -1)
        label = str((model.predict(feature_vector))[0])

        st.header('Result')
        st.text(label)
