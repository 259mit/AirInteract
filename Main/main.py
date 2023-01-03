# Libraries
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
from io import BytesIO
import plotly.express as px
import datetime
import keras
import tensorflow as tf
from streamlit_tensorboard import st_tensorboard
from keras.models import Sequential
from sklearn.decomposition import PCA
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
import base64
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import sys

# Initialize variables for capturing gestures through video camera feed.
cap = cv2.VideoCapture(0)
# Set variable to display recorded feed on the streamlit app 
FRAME_WINDOW = st.image([])
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands 
maxland = []
landmark = []
mmx = []
mmy = []
mmz = []
pin = 0

# Streamlit app UI
st.set_option('deprecation.showPyplotGlobalUse', False)
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
    
# Creating a function to process the pandas dataframe of the collected data into an XLSX data file
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

# Downloading the excel file via the streamlit app
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download csv file</a>' # decode b'abc' => abc

@st.cache(allow_output_mutation=True)

def get_data():
    return []

# Record the getsures via Google's Mediapipe engine (Pretrained model)
def record(pinsize, pin = 0):
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
  #df = pd.DataFrame({'col':maxland, 'label':label})
  return maxland

# Merge multiple recorded data frames and clean them
def prepdata(a):
    print('Merging data...')
    df = pd.concat(a)
    print('-----------------')
    k = df.columns[df.isnull().any()].tolist()  
    df = df.drop(columns = k)
    df = df.drop('Unnamed: 0', axis = 1)
    df = df.reset_index()
    print('Created dataframe: SUCCESS')
    print('-----------------')
    print('Total Null values = ', df.isnull().sum().sum())
    return df

# Data processing to plot the data
def preplot(liter, df):
    x = []
    y = []
    z = []
    labela = []
    k = [num for num in np.arange(0,21)]
    for i in k:
        x.append(df.loc[liter][i])
        y.append(df.loc[liter][str(i)+'.1']) #Appending with .1 to indicate Y axis locations
        z.append(df.loc[liter][str(i)+'.2']) #Appending with .2 to indicate Z axis locations
    labela = df.loc[liter]['label']
    return x, y, z, labela

# Plotting the getsures via plotly.
def plotges(ik, df):    
    x0, y0,z0, l0 = preplot(ik, df)
    fig = px.scatter_3d(x=x0, y=y0, z=z0)
    #fig.show()
    return fig

# Set streamlit to enable refresh/ continous data collection
@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)
    with StringIO() as buffer:
        old_write = src.write
        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)
        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield

@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield

# Model Training function
def trainmodel():
  st.title('AirInteract - Model Training Tool') # App UI
  st.header('Hand Tracking Data Collection and Annotation Toolkit') # App UI
  try:
    # Data pre-processing
    df = pd.read_excel(st.file_uploader("Upload processed Dataframe", type="xlsx"))
    x = df.drop('label', axis = 1)
    x = x.drop('Unnamed: 0', axis = 1)
    x = x.drop('index', axis = 1)
    input_shape = x.loc[0].shape
    y = df['label']
    per = st.number_input('Enter percentage of testing data points',min_value=1, max_value=50, key=None)
    st.write('X is: ')
    st.dataframe(x)
    st.markdown('Principal Component Analysis')
    usepca = st.checkbox('Use PCA ?')
    # Using principal component analysis to reduce data dimensions
    if usepca:
      try:
        n_components = st.slider('Select number of components',2, 63) # Select N components
        X = preprocessing.normalize(x)
        pca = PCA(n_components=n_components, whiten=True).fit(X)
        projected = pca.fit_transform(X)
        x = pd.DataFrame(projected)
        st.text('Scree plot') # Show scree plot
        PC_values = np.arange(pca.n_components_) + 1
        try:
          plt.figure(figsize=(6,4))
          plt.plot(PC_values, 1-pca.explained_variance_ratio_, linewidth=1)
          plt.title('% variability of components')
          plt.xlabel('Principal Component')
          plt.ylabel('Variance Captured')
          st.pyplot()
        except:
          pass
      except:
        pass
    st.write('Starting to process the data')
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=per/100, random_state=42)
    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)
    y_train = keras.utils.to_categorical(y_train, len(y.unique()))
    y_test = keras.utils.to_categorical(y_test, len(y.unique()))
    st.success('Preprocessing Done!')
    st.markdown('Building the model')
    # Add Neural Network layers
    try:
      layer1 = st.number_input('Enter number of nodes in 1st layer',min_value=16, max_value=1024, key=None) # Input node
      ka = st.number_input('Enter number of layers',min_value=1, max_value=10, key=None) 
      outl = len(y.unique())
      inact = 'relu'
      st.markdown('Choose the type of network you want to train')
      select_nn = st.selectbox('Select the type of network: ',('Simple NN','1D Convolution', 'Vanilla RNN'))
      if select_nn == 'Simple NN':
        model = Sequential([])
        model.add(Dense(layer1, activation=inact, input_shape = x.loc[0].shape))
        kp = []
        for i in range(0, ka-1):
          kp.append(st.text_input('Enter nodes, activation function for layer '+str(i+2), ''))
        for i in range(0, len(kp)):
          model.add(Dense(int(int(kp[i].split(',')[0])), activation=kp[i].split(',')[1]))
          #st.write(model.summary())
        st.success('Model built!')
        st.write('Model summary')
        with st_stdout("code"):
          model.summary()
        losst = st.selectbox('Select loss:',('Binary','Categorical'))
        if losst == 'Categorical':
          lossc = tf.losses.CategoricalCrossentropy(from_logits=True)
        optim = st.selectbox('Select optimizer:',('rmsprop','adam'))
        epo = st.number_input('Enter number of epochs',min_value=20, max_value=100, key=None)
        sttr = st.checkbox('Start Training')
        if sttr:
          model.compile(optimizer=optim, 
              loss=lossc,
              metrics=['accuracy'])
          logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
          tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
          history = model.fit(X_train, y_train,epochs=epo,callbacks=[tensorboard_callback])
          st.success('Model Trained')
          ktr = model.evaluate(X_test,  y_test, verbose=2)
          st.line_chart(history.history['accuracy'])
          st.write('Evaluation Accuracy is: ', ktr[1])
          st.markdown('---')
          st.markdown('Tensorboard: ')
          st_tensorboard(logdir=logdir, port=6006, width=700)
      if select_nn == '1D Convolution':
        st.markdown('-----')
        st.markdown('Coming Soon')
      if select_nn == 'Vanilla RNN':
        st.markdown('-----')
        st.markdown('Coming Soon')
    except:
      pass
  except:
    pass


def welcome():
  st.image(Image.open('AirInteract logo.png'))
  st.title("AirInteract - Hand Tracking Data Collection and Annotation Toolkit")
  st.header("By Mithesh Ramachandran")
  st.markdown('Light theme recommended')
  
def documentation():
  st.title("AirInteract - Documentation")
  st.image(Image.open('hand_landmarks.png'))

def process():
  st.title('AirInteract - Data Processing and Visualization Tool')
  st.header('Hand Tracking Data Collection and Annotation Toolkit')
  sheets = st.number_input('Enter number of sheet checkpoints',min_value=0, max_value=10, key=None)
  k = []
  if sheets >0:
    for i in range(0, sheets):
      try:
        k.append(pd.read_excel(st.file_uploader("Upload"+str(i), type="xlsx")))
      except:
        pass
    if k is not None:
      try:
        df = prepdata(k)
        num = st.slider('Select the datapoint',0, int(df.shape[0]-1))
        st.text('Download link:')
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
        st.plotly_chart(plotges(num, df))
      except:
        pass


def howto():
  st.title('AirInteract - Help')
  st.header('Hand Tracking Data Collection and Annotation Toolkit')
  st.subheader('How to use the Data Collection Tool: ')
  st.markdown('Record custom gestures!')
  st.markdown('1. Select the number of datapoints')
  st.text('   Datapoints will have to be constant for all recorded gestures')
  st.markdown('2. Select the label')
  st.markdown('3. Record the gesture till the frame freezes.')
  st.write('Tip: Please do not record more than 50 gestures at once, as it might use more memory. Save the checkpoint excel file and close the application. Start the application again to clear the logs.')
  st.markdown('4. Repeat by selecting the label again.')
  st.markdown('------')
  st.subheader('How to use the Data Processing and Visualization Tool: ')
  st.markdown('1. Select the number of checkpoint files needed')
  st.markdown('2. Upload the files')
  st.markdown('3. Select the data point (Observation) to view.')
  st.markdown('4. Download the processed data for model training.')

  

def tool():
  st.title('AirInteract - Data Collection Tool')
  st.header('Hand Tracking Data Collection and Annotation Toolkit')
  try:
    pinsize = st.number_input('Enter datapoints per sample',min_value=15, max_value=30, key=None)
    label = st.selectbox('Select the label',('Left', 'Right', 'Up', 'Down'))
    maxland = record(pinsize)
    #st.text(len(maxland))
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
  except:
    pass

if __name__ == "__main__":
    main()

    
    
    
                           
      
    
    
        







