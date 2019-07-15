import os
GPUS = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
from math import ceil
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.utils import class_weight
from utils import generator_batch, generator_batch_classware
from keras.utils.training_utils import multi_gpu_model

np.random.seed(1024)

FINE_TUNE = True
USE_PROCESSING =False
NEW_OPTIMIZER = True
LEARNING_RATE = 0.001
NBR_EPOCHS = 100
BATCH_SIZE = 64
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'acc'
NBR_MODELS = 250
USE_CLASS_WEIGHTS = False
RANDOM_SCALE = True
CLASS_WARE = False
nbr_gpus = len(GPUS.split(','))

train_path = './train_vehicleModel_list.txt'
val_path = './val_vehicleModel_list.txt'


if __name__ == "__main__":

    if FINE_TUNE:
        print('Finetune and Loading InceptionV3 Weights ...')
        model = load_model('./models/InceptionV3_vehicleModel-loss=0.5913-acc=0.8489-val_loss=0.4935-val_acc=0.8709.h5')

    else:
        print('Loading InceptionV3 Weights ...')
        inception = InceptionV3(include_top=False, weights= None,
               input_tensor=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling = 'avg')

        output = inception.get_layer(index = -1).output     # shape=(None, 1, 1, 2048)
        output = Dense(1024, name='features')(output)
        output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
        model = Model(outputs = output, inputs = inception.input)

    print('Training model begins...')

    optimizer = SGD(lr = LEARNING_RATE, momentum = 0.9, decay = 0.0, nesterov = True)

    if nbr_gpus > 1:
        print('Using multiple GPUS: {}\n'.format(GPUS))
        model = multi_gpu_model(model, gpus = nbr_gpus)
        BATCH_SIZE *= nbr_gpus
    else:
        print('Using a single GPU.\n')
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    #model.summary()

    # autosave best Model
    best_model_file = "./models/InceptionV3_vehicleModel.h5"
    # Define several callbacks
    best_model = ModelCheckpoint(best_model_file, monitor='val_'+monitor_index,
                                verbose = 1, save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
                  patience=3, verbose=1, min_lr=0.00001)

    early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=15, verbose=1)

    train_data_lines = open(train_path).readlines()
    # Check if image path exists.
    train_data_lines = [w for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    train_labels = [int(w.strip().split(' ')[-1]) for w in train_data_lines]
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train * 1. / BATCH_SIZE))

    val_data_lines = open(val_path).readlines()
    val_data_lines = [w for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(nbr_val))
    validation_steps = int(ceil(nbr_val * 1. / BATCH_SIZE))

    if USE_CLASS_WEIGHTS:
        print('Use Class Balanced Weights ...')
        class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    else:
        class_weights = None

    if CLASS_WARE:
        generator = generator_batch_classware
    else:
        generator = generator_batch

    model.fit_generator(generator(train_data_lines, NBR_MODELS = NBR_MODELS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                        shuffle = True, augment = True),
                        steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = generator_batch(val_data_lines,
                        NBR_MODELS = NBR_MODELS, batch_size = BATCH_SIZE,
                        img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps = validation_steps,
                        class_weight = class_weights, callbacks = [best_model, reduce_lr],
                        max_queue_size = 80, workers = 8, use_multiprocessing=True)
