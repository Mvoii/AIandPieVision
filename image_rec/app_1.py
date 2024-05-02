import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

# hide deprication warnings
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page
st.set_page_config(
    page_title = "Image Recognition",
    page_icon = "🧊",
    initial_sidebar_state = "auto"
)

# hide the part of the code, as this is just adding custom css stuff
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items(): # create dict f output classes
        if np.argmax(prediction) == clss: # check teh class
            return key # return the class name
        
with st.sidebar:
   st.image("cat_image.jpeg")
   st.title("Image Recognition")
   st.subheader("Classify the image into one of the following classes:")

# adding the model    
# st.set_option("deprecation.showfileUploaderEncoding", 0)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("cifar10_cnn_model.h5")
    return model
with st.spinner("model is being loaded... "):
    model = load_model()
st.success("Model is loaded")
    

# adding the file uploader
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) # upload the image
def import_and_predict(image_data, model):
    size = (32, 32) # set the image size
    image = ImageOps.fit(image_data, size, Image.LANCZOS) # resize the image
    img = np.asarray(image) # convert the image to array
    img_reshape = img[np.newaxis, ...] # reshape the image
    prediction = model.predict(img_reshape) # predict the image
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = 100 * np.max(predictions)
    st.sidebar.error("Accuracy : " + str(x) + "%")
    
    class_names = ["cat", "dog", "car", "truck", "airplane", "deer", "horse", "frog", "ship", "bird"]
    
    string = print("This image is most likely a " + class_names[np.argmax(predictions)] + " with a " + str(x) + "% accuracy.")


