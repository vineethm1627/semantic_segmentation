import streamlit as st
import matplotlib.pyplot as plt
import pixellib
from pixellib.semantic import semantic_segmentation
from prediction import *

#DATA_URL = "deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
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
    
    # # instantiating the semantic segmentation class
    # segment_image = semantic_segmentation()

    # # loading the model deeplabv3+ trained on pascal voc dataset.
    # segment_image.load_pascalvoc_model(DATA_URL)

    # # performing the segmentation on the input image
    # segment_image.segmentAsPascalvoc(img_file, output_image_name = "output_images/out.jpg")
    # out = plt.imread("output_images/out.jpg", 0)

    # # performing the segmentation on the input image with overlay
    # segment_image.segmentAsPascalvoc(img_file, output_image_name = "output_images/out_overlay.jpg", overlay = True)
    # out_overlay = plt.imread("output_images/out_overlay.jpg")

    out, out_overlay = prediction(img_file)
    
    col1, col2 = st.beta_columns(2)
    col1.image(out, caption = "Segmented Image", use_column_width = True)
    col2.image(out_overlay, caption = "Overlay", use_column_width = True)
