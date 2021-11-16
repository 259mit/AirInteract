# AirInteract

## A No-code one stop platform to record, annotate, visualize, merge gesture files, augment, pre-process, train models and evaluate dynamic gestures using an RGB camera.


![AirInteract logo copy](https://user-images.githubusercontent.com/64850155/141953087-30a1905b-4bb1-4039-990c-f24422d0703f.jpeg)

### By Mithesh Ramachandran


To record, pre-process, train and test the dynamic gestures, AirInteract: a no-code platform was used. AirInteract is based on Streamlit, a python-based web application deployment service. AirInteract is written completely in python. It has 3 main sections: Data Recording, Processing and Model Building.

-	The data recording tool can save recorded gesture key points, where users can pause and resume recording. Users can also annotate their data simultaneously. Users have the flexibility to change the recording duration by changing the number of datapoints.
-	The data processing tool can merge the collected datasets into one and process it, such as removing the null values, redundant columns and normalizing it. Users can also visualize the recorded gestures. Users can then download the processed data.
- The model training tool takes input of the processed dataframe, from the data processing tool. Users can split into train and test datasets by simply entering percentage of test dataset split. Users can then choose whether to use PCA. On selecting PCA, users can choose the number of components and percentage variability is then displayed. Then users can choose the type of model they wish to build. The application uses TensorFlow as the backend to build the network. First step would be to specify nodes in the input layer and entering total number of layers. Post that the user will have to just enter the number of neurons followed by the activation function. The application then displays the model summary. Users then must select the loss function and optimizer from dropdown menus and choose the number of epochs to train. Upon selecting the train checkbox, the model is trained. The training graph and the final test accuracy is displayed.
