import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
import numpy as np
import time
import cv2
import os

from tensorflow.keras import layers
from tensorflow import keras 
from PIL import Image



st.set_page_config( 
layout="wide",  
initial_sidebar_state="auto",
page_title= "MLOps CNN",
)

# function to load image
try:
    my_model = tf.keras.models.load_model('./Models/model.h5')
    face_detect = cv2.CascadeClassifier('./Models/haarcascade_frontalface_alt.xml')
    labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
except Exception:
    st.write("Error loading models")


def main():
    # Face Analysis Application #
    st.title("MLOps CNN")
    activiteis = ["Home","Face Detection"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    # C0C0C0
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face and Emotion detection.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)


    elif choice == "Face Detection":
        st.header("🔴 LIVE")
        run = st.checkbox('Run')
        time.sleep(2.0)
        def face_detection(img,size=0.5):
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              
            face_roi = face_detect.detectMultiScale(img_gray, 1.3, 1)      
            
            if face_roi == ():                                        
                return img
            
            for(x,y,w,h) in face_roi:                                
                x = x - 5
                w = w + 10
                y = y + 7
                h = h + 2
                cv2.rectangle(img, (x,y), (x+w,y+h), (125,125,10), 1)      
                img_gray_crop = img_gray[y:y+h,x:x+w]                    
                img_color_crop = img[y:y+h,x:x+w]
                
                final = cv2.resize(img_color_crop, (64,64))      
                final = np.expand_dims(final, axis = 0)     
                final = final/255.0                          

                prediction = my_model.predict(final)              
                label=labels[prediction.argmax()]                   
                cv2.putText(frame,label, (50,60), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0),3)  
                
            # fliping the image
            img_color_crop = cv2.flip(img_color_crop, 1)      

            return img
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(1)
        while run:
            ret, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('LIVE', face_detection(frame))          
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            FRAME_WINDOW.image(frame)
    else:
        pass

if __name__ == '__main__':
    main()
