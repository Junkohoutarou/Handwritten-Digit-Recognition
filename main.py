import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

st.title('Handwritten Digit Recognition')

input = open('lrc_mnist.pkl', 'rb')
model = pkl.load(input)

st.header('Upload handwritten digit image')
uploaded_file = st.file_uploader("Choose an image file", type=(['png', 'jpg', 'jpeg']))
