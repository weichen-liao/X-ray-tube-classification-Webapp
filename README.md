# X-ray-tube-classification-Webapp

### What is this Web App?
This is a Chest X-ray Catheter tube position classifier. When given a X-ray image, it predicts whether chest tubes are correctly positioned.
To be more specific, it provides probabilistic classification of chest tube positions using Machine Learning.

### The application provides classification for the following 11 labels:
ETT - Normal, ETT - Borderline, ETT - Abnormal, NGT - Abnormal
NGT - Borderline, NGT - Incompletely Image, NGT - Normal
CVC - Abnormal, CVC - Borderlin, CVC - Normal, Swan Ganz Catheter Present

![tubes](https://github.com/weichen-liao/X-ray-tube-classification-Webapp/blob/main/Tube%20Classsification.jpg)

### Interfaces
the app is intergrated with streamlit, a Python tool for machine learning desmonstration
the front page looks like this:
![frontpage](https://github.com/weichen-liao/X-ray-tube-classification-Webapp/blob/main/frontpage.png)
The funtionalities are simple and clear: upload an X-ray image from local system by clicking "Broswer files".
Read the basic information from "ABOUT".
Make the predictions and see the result from "PREDICT"

When you click the "PREIDICT", it will run the predictions on 4 different models on the backend, for each model, it will return the probability of being True for all the 11 labels. 
The final prediction will be based on the average performance of 4 models
![frontpage](https://github.com/weichen-liao/X-ray-tube-classification-Webapp/blob/main/x-ray.png)
![frontpage](https://github.com/weichen-liao/X-ray-tube-classification-Webapp/blob/main/predictions.png)
Notice the probabilities of 11 labels don't add up to 1. Because each label is predicted independently as 0 or 1.

### How it is done
The core models are trained with CNNs using transfer learning.

### What about the data
The data comes from Kaggle: https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification

Total dataset of 40,000 images, 12.23GB

As I didn't upload the models and all the other necessary files, it can't be run by you. Here you can only find the codes.

