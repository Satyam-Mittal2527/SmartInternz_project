
# Early Stage Disease Diagnosis System Using Human Nail Image

This project is build to detect the Early Disease after seeing the human nail, this proejct was given by SmartInternz were we have to form a team and complete the ptroject provided. The team of 4 was created a worked to successfully complete the project.


# About the Model

The model used was a pre-trained model ImageNet model VGG16
A convolutional neural network is also known as a ConvNet, which is a kind of artificial neural network. A convolutional neural network has an input layer, an output layer, and various hidden layers. VGG16 is a type of CNN (Convolutional Neural Network) that is considered to be one of the best computer vision models to date. The creators of this model evaluated the networks and increased the depth using an architecture with very small (3 × 3) convolution filters, which showed a significant improvement on the prior-art configurations. They pushed the depth to 16–19 weight layers making it approx — 138 trainable parameters.

https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16



## Roadmap

- Importing DataBase
- Unzip the folder and get ready with train and test Data
- Study the dataset and make the data ready for the Model(Data Preprocessing)
- Data Augmentation using ImageDataGenerator
- Importing the VGG16 Model, and make it ready to fit the data
- Fit the model in our Dataset and analyse its performance
- Get the accuracy and compare the training accuracy and Validation accuracy
- Get a new image and predict the disease
- Save the model
- Make a UI and make ready for the user(Flask -ngrok)



## Dataset

Download the Dataset

```bash
  https://drive.google.com/drive/folders/1AXTYsbiarS1TCAgfj0mancTSrJYYMWMs?usp=sharing
```
    
The Dataset Zip folder for Train and Test. Train containes 655 images and test containes 183 images with 17 classess for both.

# Installation

If you are using Google Colab
```bash
  pip install flask pyngrok --quiet
```
More Installation needed if some other platform
```bash
  pip install numpy
  pip install tensorflow
  pip install Pillow
  pip install flask
  pip install pyngrok --quiet
```

## Result/Accuracy

- Training Accuracy: 0.7009
- Validation Accuracy: 0.6562
## Demo



## 🛠️ Tech Stack

- **Python** – Core programming language
- **TensorFlow / Keras** – Deep learning and model training
- **VGG16 (ImageNet)** – Pre-trained convolutional neural network
- **Flask** – Web framework for serving the model and UI
- **Pyngrok** – Tunnels localhost for public access
- **Pillow (PIL)** – Image processing and loading
- **NumPy** – Numerical computations and data handling
- **ImageDataGenerator** – For image preprocessing and augmentation
- **Google Colabk** – Training environment


##  Source Code
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S6R-4SXhoNEuDkentDej79SLaoadV1yM)
Download the Dataset and import it in colab before running through the colab. Rename the zip files as train.zip and test.zip


## 👥 Team Members

- Satyam Mittal  https://github.com/Satyam-Mittal2527
- Progya Promita Biswas  https://github.com/Melony26
- Ekant Aman  https://github.com/EKANT8870
- Diya Baidya  https://github.com/D511200

## Working with Model

```javascript
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten,Input
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
predictions = Dense(17, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=predictions)

model.summary()


```


## Fitting the Model

```javascript
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_accuracy',
    mode= 'max',
    patience=3,
    restore_best_weights=True
)

history =model.fit(train_set,validation_data=test_set, epochs=30, steps_per_epoch = len(train_set)//3, validation_steps = len(test_set)//3)

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print("Training Accuracy: {:.4f}".format(train_acc))
print("Validation Accuracy: {:.4f}".format(val_acc))

```
