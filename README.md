
# Convolutional Neural Networks for Wildlife Species Detection

This project Demonstrates Classification of various Wildlife Species such as animals , birds and insects and predict the class of image using Deep Learning Convolutional Neural Networks.To build this model Keras library is used backend as tensorflow and building Sequential Contruction model fitting.


## Resources

 - You can find [Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) Here. This dataset contains images of 90 different wildlife animals.


## Installation of Modules

Installing required Modules

```bash
  pip install tensorflow
  pip install opencv-python
  pip install scipy
  pip install matlplotlib
  pip install numpy
```
 make sure that you installed required modules 
## Data Preparation
In this section we will split the data into training data and test data

```javascript
import os
import shutil
import random
import numpy as np
import math



os.makedirs('./data')
os.makedirs('./data/train')
os.makedirs('./data/test')

os.listdir('./data')

root_dir = './animals'

classes = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat", 
    "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", 
    "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", 
    "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", 
    "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo", 
    "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse", 
    "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", 
    "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", 
    "reindeer", "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", 
    "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", 
    "wolf", "wombat", "woodpecker", "zebra"
]

for clss in classes:
    print('------------' + clss + '-------------')
    dirtry = root_dir + '/' + clss
    files = os.listdir(dirtry)
    np.random.shuffle(files)

    base_outdir = './data/'

    for folder in ['train', 'test']:
        target_dir = base_outdir + folder
        os.makedirs(target_dir + '/' + clss)
        target_class = target_dir + '/' + clss

        if folder == 'train':
            images_to_pass = files[: math.floor(0.8*len(files))]
            for img in images_to_pass:
                img = dirtry + '/' + img
                shutil.copy(img, target_class)
        else:
            images_to_pass = files[math.floor(0.8*len(files)):]
            for img in images_to_pass:
                img = dirtry + '/' + img
                shutil.copy(img, target_class)

train_sum = 0
for animal in os.listdir('./data/train'):
    lnk = './data/train/' + animal
    train_sum += len(os.listdir(lnk))

test_sum = 0
for animal in os.listdir('./data/test'):
    lnk = './data/test/' + animal
    test_sum += len(os.listdir(lnk))

print(train_sum)
print(test_sum)
}
```
Run the above code to split the data make sure you change the directories as per your environment.

## Training your model

After splitting data into train and test set. import required modules
```javascript
import matplotlib.pyplot as plt
import cv2

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
```
Use ImageDataGenerator to perform preprocessing operations on images

```javascript
image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
                               width_shift_range=0.1, # Shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
```

```javascript
image_gen.flow_from_directory('./data/train')

image_gen.flow_from_directory('./data/test')

```
The above two scripts shows about total number of images and classes in both train and test data

Import models and layers from keras library
```javascript
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D #type: ignore

image_shape =(150,150,3)
```
Build the CNNs model. my model is Sequential model. if you want you can use different pretrained models of your convinience. you can check various models [here.](https://keras.io/api/applications/) . you can use more layers if you want.

```javascript
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, remember its multiclass
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

Checking model
```javascript
model.summary()
```

it will give 
```javascript
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 34, 34, 64)        36928     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 17, 17, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 18496)             0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               2367616   
_________________________________________________________________
activation_2 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 129       
...
Total params: 2,424,065
Trainable params: 2,424,065
Non-trainable params: 0
```
Next step is to provide Data to this model and fit the sequential model

```javascript
batch_size = 16

train_image_gen = image_gen.flow_from_directory('./data/train',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

test_image_gen = image_gen.flow_from_directory('./data/test',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

```

Next Step is to fit model.
```javascript
results = model.fit(train_image_gen,epochs=20,
                              steps_per_epoch=10,
                              validation_data=test_image_gen,
                             validation_steps=12)
```
Note: it will take time if you need more accurate better go with high ephocs and having more Dense layer or else you can use Keras Applications which are having high accuracy.

```javascript
Epoch 1/20
 9/10 [==========================>...] - ETA: 0s - loss: -626.5332 - acc: 0.0208Epoch 1/20
12/10 [====================================] - 5s 457ms/step - loss: -661.0068 - acc: 0.0104
10/10 [==============================] - 11s 1s/step - loss: -634.4128 - acc: 0.0188 - val_loss: -642.1359 - val_acc: 0.0104
Epoch 2/20
 9/10 [==========================>...] - ETA: 0s - loss: -703.4123 - acc: 0.0139Epoch 1/20
12/10 [====================================] - 5s 412ms/step - loss: -661.0068 - acc: 0.0104
10/10 [==============================] - 10s 981ms/step - loss: -706.9581 - acc: 0.0125 - val_loss: -642.1359 - val_acc: 0.0104
Epoch 3/20
 9/10 [==========================>...] - ETA: 0s - loss: -732.2686 - acc: 0.0139Epoch 1/20
12/10 [====================================] - 6s 504ms/step - loss: -661.0068 - acc: 0.0104
10/10 [==============================] - 12s 1s/step - loss: -722.8664 - acc: 0.0125 - val_loss: -642.1359 - val_acc: 0.0104
Epoch 4/20
 9/10 [==========================>...] - ETA: 0s - loss: -665.0792 - acc: 0.0000e+00Epoch 1/20
12/10 [====================================] - 6s 460ms/step - loss: -661.0068 - acc: 0.0104
10/10 [==============================] - 11s 1s/step - loss: -670.9250 - acc: 0.0000e+00 - val_loss: -642.1359 - val_acc: 0.0104
Epoch 5/20
 9/10 [==========================>...] - ETA: 0s - loss: -718.6391 - acc: 0.0069Epoch 1/20
12/10 [====================================] - 5s 406ms/step - loss: -661.0068 - acc: 0.0104
10/10 [==============================] - 11s 1s/step - loss: -710.1206 - acc: 0.0125 - val_loss: -642.1359 - val_acc: 0.0104
Epoch 6/20
 9/10 [==========================>...] - ETA: 0s - loss: -716.6159 - acc: 0.0139Epoch 1/20
12/10 [====================================] - 5s 431ms/step - loss: -661.0068 - acc: 0.0104
10/10 [==============================] - 10s 1s/step - loss: -727.8497 - acc: 0.0125 - val_loss: -642.1359 - val_acc: 0.0104
Epoch 7/20
...
Epoch 20/20
 9/10 [==========================>...] - ETA: 0s - loss: -723.5372 - acc: 0.0139Epoch 1/20
12/10 [====================================] - 5s 425ms/step - loss: -661.0068 - acc: 0.0104
10/10 [==============================] - 10s 1s/step - loss: -717.7872 - acc: 0.0125 - val_loss: -642.1359 - val_acc: 0.0104
```
In my case i used less ephochs and layers. if you want to fine tune your model you need to increase ephocs and hidden layers.

You can save this model and load it later for prediction purpose.
```javascript
model.save('wildlife.h5')
```

Evaluating model

Load the Image and preprocess it

```javascript
img = cv2.imread('/your_image_path') ### laoding image

img = image.imload(img,resize_shape=(150,150)) ### reshaping image it should same shape as when you fit your model

x = image.img_to_array(img)  ### Converting image to array


x = x/255   
x = np.expand_dims(x, axis = 0)
img_data = preprocess_input(x)

```

Predict the model

```javascript
model.predict(img_data)

a = np.argmax(model.predict(img_data), axis = 1)

```
the a returns the index of class. if you want to see indices list

```javascript
train_image_gen.class_indices

{'antelope': 0,
 'badger': 1,
 'bat': 2,
 'bear': 3,
 'bee': 4,
 'beetle': 5,
 'bison': 6,
 'boar': 7,
 'butterfly': 8,
 'cat': 9,
 'caterpillar': 10,
 'chimpanzee': 11,
 'cockroach': 12,
 'cow': 13,
 'coyote': 14,
 'crab': 15,
 'crow': 16,
 'deer': 17,
 'dog': 18,
 'dolphin': 19,
 'donkey': 20,
 'dragonfly': 21,
 'duck': 22,
 'eagle': 23,
 'elephant': 24,
...
 'whale': 85,
 'wolf': 86,
 'wombat': 87,
 'woodpecker': 88,
 'zebra': 89}
```
you can postprocess this to draw spcies text and draw text on objects


## Disclaimer

You can find the full code for this model in ipynb format. you can go through the steps or else you can load in text editor like Visual Studio Code.



## Feedback

If you have any feedback, please reach out to me at hsherlock366@gmail.com. Thanks you for Coming Here.

