# Lipreading_DNN
A CNN-LSTM Deep Neural Network for Lipreading.

The model is intended to be used with the LRS3 dataset. Which can be requested here: http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html

The model must first be preprocessed, using process_data.py. Which requires the dlib ""shape_predictor_68_face_landmarks.dat" predictor, found here: https://github.com/davisking/dlib-models
Once the data has been processed it can be used to train the Lipreading model, by running train_model.py.

Using 600 samples per category, the model has achieved 94.48% across 10 different words.
