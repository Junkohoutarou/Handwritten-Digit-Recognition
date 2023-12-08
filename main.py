import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

IMG_SIZE = 227

class_list = {'0': 'NORMAL', '1': 'PNEUMONIA'}

st.title('Pneumonia prediction based on chest X-Ray image')

input = open('lrc_xray.pkl', 'rb')
model = pkl.load(input)

st.header('Upload hardwritten digit image')
uploaded_file = st.file_uploader("Choose an image file", type=(['png', 'jpg', 'jpeg']))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Test image')

    if st.button('Predict'):
        image = image.resize((IMG_SIZE*IMG_SIZE*3, 1))
        feature_vector = np.array(image)
        label = str((model.predict(feature_vector))[0])

        st.header('Result')
        st.text(class_list[label])
