import pixellib
import matplotlib.pyplot as plt
from pixellib.semantic import semantic_segmentation
DATA_URL = "deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"

def prediction(img_file):
    # instantiating the semantic segmentation class
    segment_image = semantic_segmentation()

    # loading the model deeplabv3+ trained on pascal voc dataset.
    segment_image.load_pascalvoc_model(DATA_URL)

    # performing the segmentation on the input image
    segment_image.segmentAsPascalvoc(img_file, output_image_name = "output_images/out.jpg")
    out = plt.imread("output_images/out.jpg", 0)

    # performing the segmentation on the input image with overlay
    segment_image.segmentAsPascalvoc(img_file, output_image_name = "output_images/out_overlay.jpg", overlay = True)
    out_overlay = plt.imread("output_images/out_overlay.jpg")

    return out, out_overlay
