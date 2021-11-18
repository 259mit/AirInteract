# Libraries
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
import os
from io import BytesIO
import cv2
import base64
import pickle
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands 
#run = st.sidebar.checkbox('Run')
#stop = st.sidebar.checkbox('Stop')
#display = st.checkbox('Display')
FRAME_WINDOW = st.image([])
pin = 0
max = []
mmx = []
mmy = []
mmz = []
maxland = []
landmark = []
landx = []
landy = []
landz = []
cap = cv2.VideoCapture(0)
pin = 0
num = 0
da = pd.DataFrame()
df = pd.DataFrame()

@st.cache(allow_output_mutation=True)

def get_data():
    return []

def welcome():
  st.image(Image.open('AirInteract logo.png'))
  st.title("AirInteract - Hand Tracking Data Collection and Annotation Toolkit")
  st.header("By Mithesh Ramachandran")

def trainmodel():
  a = 'hi'
  return a

def howto():
  a = 'hi'
  return a


def process():
  a = 'hi'
  return a

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def documentation():
  a = 'hi'
  return a

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download csv file</a>' # decode b'abc' => abc


def tool():
    pin = 0
    st.title('AirInteract - Data Collection Tool')
    st.header('Hand Tracking Data Collection and Annotation Toolkit')
    pinsize = st.number_input('Enter datapoints per sample',min_value=10, max_value=30, key=None)
    customlabel = st.checkbox('Custom Label')
    if customlabel:
      sl = st.text_input('Enter Labels followed by comma and space', '')
      kl = tuple(sl.split(', '))
      label = st.selectbox('Select the label',kl)
    else:
      label = st.selectbox('Select the label',('Left', 'Right', 'Up', 'Down'))
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      while pin<pinsize:
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks is not None:
          for hand_landmarks in results.multi_hand_landmarks:
                landmark.append(hand_landmarks.landmark)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()) 
                #st.text(landmark)
                #maxland.append(landmark)
          #df = pd.DataFrame({'col':landmark, 'label':label})
          pin = pin + 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(image) 
    maxland.append(landmark)
    for i in range(0, len(maxland)):
        for j in range(0, len(maxland[i])):
            mmx.append(maxland[i][j][8].x)
            mmy.append(maxland[i][j][8].y)
            mmz.append(maxland[i][j][8].z)
    get_data().append({'all': maxland, 'x': mmx, 'y': mmy, 'z': mmz, 'Labels': label})    
    k = pd.DataFrame(get_data())
    dfx = k.x.apply(pd.Series)
    dfy = k.y.apply(pd.Series)
    dfz = k.y.apply(pd.Series)
    dd = pd.concat([dfx, dfy], axis=1)
    data = pd.concat([dd, dfz], axis=1)
    data['label'] = k['Labels']
    st.write(data)
    st.text('Download link:')
    st.markdown(get_table_download_link(data), unsafe_allow_html=True)

def main():
    st.sidebar.image(Image.open('AirInteract logo.png'), width=300)
    st.sidebar.title('AirInteract')
    st.sidebar.header('Main menu')
    selected_box = st.sidebar.selectbox(
    'Select the page: ',
    ('Welcome','Data Collection Tool', 'Data Processing Tool','Model Training Tool','Help','Documentation')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Data Collection Tool':
        tool()
    if selected_box == 'Data Processing Tool':
        process()
    if selected_box == 'Model Training Tool':
        trainmodel()
    if selected_box == 'Help':
        howto()
    if selected_box == 'Documentation':
        documentation()
  

if __name__ == "__main__":
    main()
    
    
                           
      
    
    
        







