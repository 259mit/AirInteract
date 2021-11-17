# AirInteract

A No-code one stop platform to record, annotate, visualize, merge gesture files, augment, pre-process, train models and evaluate dynamic gestures using an RGB camera.


![AirInteract logo copy](https://user-images.githubusercontent.com/64850155/141953087-30a1905b-4bb1-4039-990c-f24422d0703f.jpeg)

### By Mithesh Ramachandran


To record, pre-process, train and test the dynamic gestures, introducing a no-code platform: AirInteract. AirInteract is based on Streamlit, a python-based web application deployment service. AirInteract is written completely in python. It has 3 main sections: Data Recording, Processing and Model Building.

-	The data recording tool can save recorded gesture key points, where users can pause and resume recording. Users can also annotate their data simultaneously. Users have the flexibility to change the recording duration by changing the number of datapoints.
-	The data processing tool can merge the collected datasets into one and process it, such as removing the null values, redundant columns and normalizing it. Users can also visualize the recorded gestures. Users can then download the processed data.
- The model training tool takes input of the processed dataframe, from the data processing tool. Users can split into train and test datasets by simply entering percentage of test dataset split. Users can then choose whether to use PCA. On selecting PCA, users can choose the number of components and percentage variability is then displayed. Then users can choose the type of model they wish to build. The application uses TensorFlow as the backend to build the network. First step would be to specify nodes in the input layer and entering total number of layers. Post that the user will have to just enter the number of neurons followed by the activation function. The application then displays the model summary. Users then must select the loss function and optimizer from dropdown menus and choose the number of epochs to train. Upon selecting the train checkbox, the model is trained. The training graph and the final test accuracy is displayed.

## The App in action

https://user-images.githubusercontent.com/64850155/141962733-33931c61-f1c2-42e8-902d-82b4ffa3ec77.mov

___________________

The Home tab
---
<img width="1400" alt="Screenshot 2021-10-11 at 4 31 43 PM" src="https://user-images.githubusercontent.com/64850155/141954421-435b796a-49f9-4a60-931c-a2b43e9ae8bf.png">

```
The home screen of Airinteract has the navigation section, 
where users can navigate to the page of their choice.
```
___________________

The Recording tab
---
<img width="1400" alt="Screenshot 2021-10-13 at 3 55 24 PM" src="https://user-images.githubusercontent.com/64850155/141966266-77dddd59-85ba-4a5e-82fc-6c89e7959077.png">

```
The Recording screen of Airinteract, 
where users can record their gestures.
```

<img width="1400" alt="Screenshot 2021-11-16 at 2 34 30 PM" src="https://user-images.githubusercontent.com/64850155/141987912-4a0ceacf-1f13-4b08-9056-cd3316f86196.png">

```
The Recording screen of Airinteract, 
where users can select the type of the gestures they want.
```

<img width="1400" alt="Screenshot 2021-11-16 at 2 35 48 PM" src="https://user-images.githubusercontent.com/64850155/141988047-d24e6bf6-7ef9-4453-a192-6bf6389c1981.png">

```
The Recording screen of Airinteract, 
where users can annotate and add custom annotation labels
and save checkpoints as csv files.
```

___________________

The Data Processing and Visualization tab
---

https://user-images.githubusercontent.com/64850155/141989149-cf16a5fc-3bd4-4855-ac9d-18b09b0e4a02.mov

```
The Data Processing and Visualization screen of Airinteract, 
where users can merge gesture csv checkpoint files and pre-process them.
They can also scroll through samples and visualize them.
They can also save the merged file.
```

___________________

The Model Training tab
---

https://user-images.githubusercontent.com/64850155/142181711-60986399-7635-4358-8bd3-de324fdc4f67.mov


```
The model building and training screen of Airinteract, 
where users can view the training predictor files.
They can also use techniques to reduce dimensionality.
They can select the type of network they wish to train.
They can set and tune hyperparameters
And also evaluate the model
```
