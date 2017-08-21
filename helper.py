import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

FULL_PATH ="/Users/stefanmarwah/Desktop/SDCarND/SDCarND-Term1/Term_1_Projects/term_1_git_repo/P3-BehavioralCloning/CarND-Behavioral-Cloning-P3-master/data/IMG/"
LOG_FILES = ["data/track{}_clockwise_driving_log_{}.csv","data/track{}_anti_clockwise_driving_log_{}.csv"]
IMAGE_PATHS = ["data/TRACK{}_IMG_CLOCKWISE_{}/","data/TRACK{}_IMG_ANTI_CLOCKWISE_{}/"]
NUM_ROUNDS = 2

def replace_image_file_path(log_file,replace_from,image_path,write_to):
    """
    log_file: The name of the log file which contains the driving log data.
    replce_from: The path name to replace.
    image_path: The actual image path to replcae to.
    write_to: The name of the file to write to.

    The goal of this function is to replace the path to the image in the driveing log file
    """
    f = open(log_file)
    w = open(write_to,'a+')
    for line in f:
        row = line.split(',')
        row_0 = image_path+row[0][len(replace_from):]
        row_1 = image_path+row[1][len(replace_from):]
        row_2 = image_path+row[1][len(replace_from):]
        w.write((',').join([row_0,row_1,row_2] + row[3:]))
    w.close()
    f.close()

def combine_all_data(log_files,image_paths,write_to):
    """
    log_files: List of log file names
    image_paths: List of image paths names
    write_to: Name of the file to write to

    The goal of thsi function to combine the data from all the driving log files into one file
    """

    with open(write_to, "w"):
        pass
    for i in range(NUM_ROUNDS-1):
        for log_file, image_path in zip(log_files,image_paths):
            replace_image_file_path(log_file.format(i+1,i),FULL_PATH,image_path.format(i+1,i),write_to)
            replace_image_file_path(log_file.format(i+1,i+1),FULL_PATH,image_path.format(i+1,i+1),write_to)

    for i in range(1,NUM_ROUNDS):
        for log_file, image_path in zip(log_files,image_paths):
            replace_image_file_path(log_file.format(i+1,i-1),FULL_PATH,image_path.format(i+1,i-1),write_to)
            replace_image_file_path(log_file.format(i+1,i),FULL_PATH,image_path.format(i+1,i),write_to)

    for i in range(2,NUM_ROUNDS+1):
        for log_file, image_path in zip([log_files[1]],[image_paths[1]]):
            replace_image_file_path(log_file.format(i-1,i),FULL_PATH,image_path.format(i-1,i),write_to)
            replace_image_file_path(log_file.format(i,i),FULL_PATH,image_path.format(i,i),write_to)


def filter_steering_angle(log_file,write_to_file,steering_angle_threshold):
    """
    log_file: The name of the driving log file.
    write_to: The name of the file to write to.
    steering_angle_threshold: The steering value above which the data is to be filtered

    The goal of this function is to filter the data that is above the steering threshold value
    """
    write_to = open(write_to_file,'w')
    with open(log_file) as csvfile:
        for line in csvfile:
            row = line.split(',')
            steering_angle = float(row[3])
            if steering_angle <= -steering_angle_threshold or steering_angle > steering_angle_threshold:
                write_to.write(line)
    write_to.close()

def generator(samples,batch_size=32):
    """
    samples: The sample of data that is to be preprocessed.
    batch_size: The batch size of the sample that is to be preprocessed.

    The goal of this function is to continously return batches of the data
    and process them on the fly only when you needed and not all at once.
    """
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                measurement = float(batch_sample[3])
                image_path = batch_sample[0]
                image = cv2.imread(image_path)
                images.append(image)

                measurements.append(measurement)

                augmented_image = cv2.flip(image,1)
                images.append(augmented_image)
                augmented_measurement = measurement*-1.0
                measurements.append(augmented_measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train,y_train)

def get_samples(from_file):
    """
    from_file: The name of the file, to retrieve the sample data from.

    The goal of this function is to retrieve the sample data from the specified file.
    """
    samples = []
    with open(from_file) as csvfile:
        reader =csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def get_test_data(samples):
    X = []
    y = []
    for sample in samples:
        X.append(cv2.imread(sample[0]))
        y.append(float(sample[3]))
    return np.array(X),np.array(y)

def plot_loss(loss,val_loss,nb_epoch,file_name):
    epochs = [x for x in range(nb_epoch)]
    plt.plot(epochs,loss,label='Train loss')
    plt.plot(epochs,val_loss,label='Validation loss')
    plt.legend(loc=1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss Vs Validation Loss')
    plt.savefig("examples/"+file_name)
