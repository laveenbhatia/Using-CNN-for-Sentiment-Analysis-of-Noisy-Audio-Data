# Using CNN for Sentiment Analysis of noisy audio data


## Introduction
Sentiment Analysis, refers to the use of various NLP techniques to identify the sentiment of a given data. This data can be in one of the three formats - audio/video/text. In this project, we use CNN to perform sentiment analysis of noisy audio data. The main aim of the model will be to correctly clasify an audio signal into one of the three classes - negative/neutral/positive.

## Dataset
The dataset used in this project is [CMU-MOSI](https://github.com/A2Zadeh/CMU-MultimodalSDK) dataset, which is a collection of 2199 opinion audio clips. Each opinion audio is annotated with sentiment in the range [-3,3]. Since the labels in the dataset are continuious rather than categorical, I have used the following throsholds to convert this data into 3 categories.
* Negative - [-3 to < -0.5]
* Neutral - [-0.5 to 0.5]
* Positive - [>0.5 to 3]  
The audio data can be found in the following subdirectory of this repo - main/data/Audio

## Preprocessing the data
#### Adding the noise
To add the noise, we first create a numpy array of random normal distribution with mean of 0, standard deviation equal to that of the signal and length also equal to that of signal. Then we multiply this noise array with a noise_factor which is in the range of (0, 1). This decides the percentage of noise which will be added to the signal. Finally we add this noise to the audio signal.
#### Extracting the mel spectrogram
Then, we extract the mel spectrogram of the audio signals and save in a separate folder.

The entire code for the preprocessing of data can be found in the following file - main/src/data_preprocess.py

## Training the model
To train the model, use the Sequential() class of tensorflow.keras library. In this sequential model, we add our Convolutional/Pooling layers as required.
The code for the training of the model can be found in - main/src/model_train.py

## Evaluation the model
Evaluation of the model is done using the following matrices:
* Accuracy
* Auc Score
* ROC-AUC Curve
The code for the evaluation can be found in - main/src/model_evaluate.py

## Instructions to run the project
* If you wish to perform a quick run of the project then just use [quickrun.ipynb](quickrun.ipynb) and run it in a jupyter notebeook. There is no need to perform the following steps. If you wish to train the model locally then follow following steps.
* Run the folloiwng command to clone the github repo
```bash
git clone https://github.com/laveenbhatia/Using-CNN-for-Sentiment-Analysis-of-Noisy-Audio-Data.git
```

* Run the following command to install the required dependencies
```bash
pip install -r requirements.txt
```

* If you wish to use your own custom data to train then follow following steps, otherwise if you wish to just train the model using the preprocessed data cloned from repo, skip directly to the next point.
   * Go to data/Audio - Inside this folder you need to have one folder for each class. For example, in my project I am using 3 classes - Negative, Neutral and Positive. So, you will find 3 folder with the name of each class. Similarly, you need to have one folder for each class you have in your dataset.
   * Run data_preprocess.py in main/src
* To train the model, run model_train.py in main/src. In this file, if you wish you can fine tune the various hyperparameters such as number of layers, activation functions, etc.
* To evaluate the model, run model_evaluate.py in main/src.