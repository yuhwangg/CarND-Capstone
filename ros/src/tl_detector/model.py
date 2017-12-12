from keras.utils import  to_categorical
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, MaxPool2D, BatchNormalization, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
import datetime
import numpy as np
import cv2
import csv
import math
import matplotlib.pyplot as plt

models_path = './models/'

path_dataset_img = 'traffic_light_bag_files/dataset/'
path_dataset_index = 'traffic_light_bag_files/dataset/dataset.csv'
default_batch_size = 32
default_split_valid = 0.2

def loadDataset(path,split):
    rows = []
    with open(path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            rows.append(row)
    # sklearn.utils.shuffle not available, done using numpy
    np.random.shuffle(rows)

    paths = []
    y = []
    for row in rows:
        paths.append(row[0])
        y.append(row[1])

    # sklearn.model_selection.train_test_split not available in VM, do it manually :(
    size = len(paths)
    size_valid = (int)(size * split)
    size_train = size - size_valid
    paths_train, paths_valid = paths[:size_train], paths[size_train:]
    y_train, y_valid = y[:size_train], y[size_train:]
    #print(len(paths_train), len(paths_valid))
    #print(len(paths_train) + len(paths_valid), '=', size)
    dataset_train = {"path": paths_train, "y": y_train}
    dataset_valid = {"path": paths_valid, "y": y_valid}
    return dataset_train, dataset_valid

def loadDatasetGenerators(dataset_path, batch_size=default_batch_size, split=default_split_valid):  # return generator
    basepath = dataset_path
    train_dataset, valid_dataset = loadDataset(basepath,split=split)

    train_size = len(train_dataset['path'])
    valid_size = len(valid_dataset['path'])

    sample = cv2.imread(path_dataset_img + train_dataset['path'][0])
    sample_shape = sample.shape
    sample_type = type(sample[0][0][0])

    info = {
        'n_train':train_size,
        'n_train_batch': math.ceil(train_size/batch_size),
        'n_valid':valid_size,
        'n_valid_batch': math.ceil(valid_size/batch_size),
        'input_shape': sample_shape,
        'data_type': sample_type
    }

    return datasetGenerator(train_dataset, batch_size, augment=False), datasetGenerator(valid_dataset, batch_size), info

def augmentImage(img):
    #TODO: placeholder for img augmentation, if needed
    return img

def datasetGenerator(dataset, batch_size=default_batch_size, augment=False):
    n_dataset = len(dataset)
    paths = dataset['path']
    labels = dataset['y']

    while 1:
        #paths, labels = shuffle(paths, labels)
        for offset in range(0, n_dataset, batch_size):
            batch_paths = paths[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]
            batch_labels_one_hot = to_categorical(batch_labels, num_classes=4)
            X = []
            y = []
            for path,label in zip(batch_paths, batch_labels_one_hot):
                img = cv2.imread(path_dataset_img+path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #opencv loads in BGR
                if augment:
                    img = augmentImage(img)

                X.append(img)
                y.append(label)

            X = np.array(X)
            y = np.array(y)
            yield (X,y)

def standardDatetime():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def create_model(input_shape, name='model', load_weights=None, debug=False):

    model = Sequential(name=name)
    model.add(Lambda(lambda x: x/255, input_shape=input_shape))  # normalize
    model.add(Conv2D(filters=20, kernel_size=(3, 3), activation='relu', name='conv_0', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=40, kernel_size=(3, 3), activation='relu', name='conv_1', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=80, kernel_size=(3, 3), activation='relu', name='conv_2', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=80, kernel_size=(3, 3), activation='relu', name='conv_3', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(50, activation="relu", name='fc0'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax', name='fc1'))

    if load_weights is not None:
        print('Loading weights', load_weights)
        model.load_weights(load_weights)
    else:
        print('Loading weights failed', load_weights)

    if debug:
        model_img_path = models_path + model.name + '.png'
        plot_model(model, to_file=model_img_path, show_shapes=True)

    return model

def train(dataset, epochs=30, batch_size=32, load_weights=None, debug=False):
    timestamp = standardDatetime()
    # load dataset generator and metrics
    gen_train, gen_valid, info = loadDatasetGenerators(dataset, batch_size=batch_size)
    print(info)

    # create the model a eventually preload the weights (set to None or remove to disable)
    model = create_model(input_shape=info['input_shape'], load_weights=load_weights, debug=debug)
    model_name = model.name + '_' + timestamp

    # Intermediate model filename template

    filepath = models_path + model_name + "_{epoch:02d}-{val_loss:.5f}-{val_acc:.5f}.h5"
    # save model after every epoc, only if improved the val_loss.
    # very handy (with a proper environment (2 GPUs anywhere) you can test your model while it still train)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    # detect stop in gain on val_loss between epocs and terminate early, avoiding unnecessary computation cycles.
    earlystopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1)
    callbacks_list = [checkpoint, earlystopping]

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    history_object = model.fit_generator(gen_train, info['n_train_batch'], verbose=1, epochs=epochs,
                                         validation_data=gen_valid, validation_steps=info['n_valid_batch'],
                                         callbacks=callbacks_list)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['acc'])
    plt.plot(history_object.history['val_acc'])
    plt.title('model mean squared error accuracy')
    plt.ylabel('mean squared error accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper right')
    plt.show()

    return model_name

if __name__ == '__main__':
    train(path_dataset_index)