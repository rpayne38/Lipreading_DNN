#Program to train a CNN-LSTM network, using training data from the LRS3 dataset, preprocessed using process_data.py
#The CNN used is VGG16, with ImageNet weights.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TimeDistributed, LSTM, Masking, Dropout

import os
import numpy as np
import cv2
from random import shuffle

print(tf.__version__)

MAX_FRAMES=15
train_path="train"
val_path="val"
WORDS=["SOMETHING", "THEN", "WORLD", "BECAUSE", "YOU", "PEOPLE", "REALLY", "ABOUT", "LIKE", "WE"]
epochs=80

#gets shuffled list of videos in input folder
def sortFiles(input):
    listOfFiles=list()
    listOfVideos=list()
    for (dirpath, dirnames, filenames) in os.walk(input):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    for file in listOfFiles:
        if file.endswith(".mp4"):
            listOfVideos.append(file)

    shuffle(listOfVideos)
    return listOfVideos

#returns numpy array of frames and labels
def training_gen():
    files=sortFiles(train_path)
    for file in files:
        base=os.path.basename(file)
        label,num=base.split("_")
        label=WORDS.index(label)
        label=tf.one_hot(label,len(WORDS))

        cap=cv2.VideoCapture(file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pad=MAX_FRAMES-frameCount
        buf = np.empty((frameCount, 224, 224, 3), np.dtype('float32'))

        fc=0
        ret=True

        while (fc < frameCount  and ret):
            ret, buf[fc] = cap.read()
            fc += 1
        
        cap.release()
        yield (np.pad(buf,((0,pad),(0,0),(0,0),(0,0)),'constant', constant_values=(0.0))),label

#returns numpy array of frames and labels
def validation_gen():
    files=sortFiles(val_path)
    for file in files:
        base=os.path.basename(file)
        label,num=base.split("_")
        label=WORDS.index(label)
        label=tf.one_hot(label,len(WORDS))

        cap=cv2.VideoCapture(file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pad=MAX_FRAMES-frameCount
        buf = np.empty((frameCount, 224, 224, 3), np.dtype('float32'))

        fc=0
        ret=True

        while (fc < frameCount  and ret):
            ret, buf[fc] = cap.read()
            fc += 1
        
        cap.release()
        yield (np.pad(buf,((0,pad),(0,0),(0,0),(0,0)),'constant', constant_values=(0.0))),label

#generate train and validation datasets
output_shape=(tf.TensorShape([MAX_FRAMES,224,224,3]), tf.TensorShape([len(WORDS)]))
train=tf.data.Dataset.from_generator(training_gen, output_types=(tf.float32,tf.float32),output_shapes=output_shape).batch(4)
val=tf.data.Dataset.from_generator(validation_gen, output_types=(tf.float32,tf.float32),output_shapes=output_shape).batch(4)

# for x,y in train:
#     print(x,y)
#     exit()

#load pretrained CNN
vgg_model=tf.keras.applications.VGG16(include_top=True, weights='imagenet')
vgg_model.trainable=False

cnn=Sequential()
for layer in vgg_model.layers[:-2]:
    cnn.add(layer)

#Add LSTM and Dense layers to CNN
model=Sequential()
model.add(tf.keras.layers.Masking(mask_value=0.0,input_shape=(MAX_FRAMES,224,224,3)))
model.add(TimeDistributed(cnn, input_shape=(MAX_FRAMES,224,224,3)))
model.add(LSTM(2048,activation="relu"))
tf.keras.layers.Dropout(0.1),
model.add(Dense(2048, activation="relu"))
tf.keras.layers.Dropout(0.1),
model.add(Dense(2048, activation="relu"))
tf.keras.layers.Dropout(0.1),
model.add(Dense(len(WORDS),activation="softmax"))

model.summary()

#create callbacks
log_dir = "logs/fit/Lipreading_1"
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "cp-{epoch:03d}.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 10 epochs
cb = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=2, period=10),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=["accuracy"])

history=model.fit(train,
    epochs=epochs, 
    validation_data=val,
    verbose=1,
    callbacks=cb,
    shuffle=False)

model.save("Lipreading.h5")
