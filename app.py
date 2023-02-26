import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
import numpy as np
import time
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image,ImageOps

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
#PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(FILE_DIR, "resources")
IMAGE_PATH = os.path.join(dir_of_interest, "images", "potato_image.jpg")
MODEL_PATH = os.path.join(dir_of_interest, "model", "potato1")
image = Image.open(IMAGE_PATH)

def load_model():
    model=tf.keras.models.load_model(MODEL_PATH)
    return model

with st.sidebar: 
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Home","Detect"],
        icons = ["house","app"],
        menu_icon = "cast",
        default_index= 0,
    )

if selected == "Home":
    st.title(':red[Potato leaf Disease Classification]')
    st.subheader(':blue[By using this web application you can detect the dieases of Potato leaf.]')
    st.write('The model is build to detect three different disease of potato leaf.')
    st.markdown('- Potato Early blight')
    st.markdown('- Potato Late blight')
    st.markdown('- Potato healthy')
    st.image(image)
    st.write('click on Detect to check the disease of potato leaf')

if selected == "Detect":
    model = load_model()
    st.header(':green[Potatao Leaf Disease Classification]')
    file = st.file_uploader('Please upload a Leaf',type=['jpg','png'])

    def import_and_predict(image_data,model):
        size = (256,256)
        image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
        img =np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)

        return prediction
    
    if file is None:
        st.text('Please upload an image file')
    else:
        image = Image.open(file)
        st.image(image,use_column_width= True)
        predictions = import_and_predict(image,model)
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        classes = {'Potato___Early_blight':'Potato Early blight', 'Potato___Late_blight' : 'Potato Late blight', 'Potato___healthy': 'No disease. Leaf is healthy'}
        confidence = np.max(predictions)
        predict = class_names[np.argmax(predictions)]
        confident = round(confidence*100,2)
        disease = f'Disease: :blue[{classes[predict]}]'
        confidence = f'Confidence: :blue[{confident}%]'
        with st.spinner('Wait for it...'):
            time.sleep(1)

        st.write(disease)
        st.write(confidence)
        
        string_=f"Detection completed"
        st.success(string_)