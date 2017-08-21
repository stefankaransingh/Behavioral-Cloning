#Required Libraries
import numpy as np

#Data Wrangling
from helper import *
FULL_PATH ="/Users/stefanmarwah/Desktop/SDCarND/SDCarND-Term1/Term_1_Projects/term_1_git_repo/P3-BehavioralCloning/CarND-Behavioral-Cloning-P3-master/data/IMG/"
LOG_FILES = ["data/track{}_clockwise_driving_log_{}.csv","data/track{}_anti_clockwise_driving_log_{}.csv"]
IMAGE_PATHS = ["data/TRACK{}_IMG_CLOCKWISE_{}/","data/TRACK{}_IMG_ANTI_CLOCKWISE_{}/"]
COMPLETE_DRIVE_LOG = "complete_log_data.csv"
FILTERED_LOG = "filtered_log_data.csv"
STEERING_ANGLE_THRESHOLD=0.75
combine_all_data(LOG_FILES,IMAGE_PATHS,COMPLETE_DRIVE_LOG)
filter_steering_angle(COMPLETE_DRIVE_LOG,FILTERED_LOG,STEERING_ANGLE_THRESHOLD)

#Splitting the data into training and vlaidation datasets
from sklearn.model_selection import train_test_split
samples = get_samples(FILTERED_LOG)
train_samples, test_samples = train_test_split(samples, test_size=0.2)
validation_samples,test_samples =train_test_split(test_samples,test_size=0.1)
train_generator = generator(train_samples,batch_size=16)
validation_generator = generator(validation_samples,batch_size=16)

#Define the Model

#Required Libraries for defining and training the model
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Convolution2D,Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error

#The Model Architecture
def Model():
    model = Sequential()

    #Normalize the image
    model.add(Lambda(lambda x:x /255.0 - 0.5,input_shape=(160,320,3)))

    #Crop each image, to only include the region of interest (i.e. only the road)
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam')

    return model

#Define the Model
model = Model()

#Model parameters
EPOCHS = 30

# checkpoint
filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

History = model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples =len(validation_samples),
                    nb_epoch=EPOCHS,
                    callbacks=callbacks_list)

plot_loss(History.history['loss'],History.history['val_loss'],EPOCHS,'loss')

X_test,y_test =get_test_data(test_samples)
y_pred = model.predict(X_test)
print("The mean squared error on the test data is {}".format(mean_squared_error(y_test,y_pred)))
