import streamlit as st
import fyp
from PIL import Image

st.header("""Annotating System on Musical Score Sheet""")
st.sidebar.header('Parameter')
sentence = st.sidebar.slider("Please select the number of sentences before uploading", 1, 10, 4)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
	image =Image.open(uploaded_file)
	fyp.openImage(image, sentence)
	image = Image.open('Copy23.png')
	st.image(image, caption='Annotated image')







