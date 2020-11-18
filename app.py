import streamlit as st
import matplotlib.pyplot as plt
import pixellib
from pixellib.semantic import semantic_segmentation


st.title("Sematic Image Segmentation")
st.sidebar.title("Semantic Image Segmentation")
st.sidebar.markdown("deeplabv3+ trained on pascal voc dataset for Semantic Image Segmentation using PixelLib. Here, Different instances of the same object are segmented with the same color map.")
pic = plt.imread("xception_model_colormap.png")
st.sidebar.image(pic, caption = "Model classes", use_column_width = True)
st.set_option('deprecation.showfileUploaderEncoding', False)
img_file = st.file_uploader("Upload the input image : ", type = ['jpg', 'jpeg', 'png'])

if img_file is not None:
    img = plt.imread(img_file, 0)
    st.image(img, caption = "Input Image", use_column_width = True)
    
    
