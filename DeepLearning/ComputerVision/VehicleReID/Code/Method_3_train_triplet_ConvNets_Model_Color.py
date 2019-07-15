import os
GPUS = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
from math import ceil
import numpy as np
import copy
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Input, concatenate, subtract, dot, Activation, add, merge, Lambda
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD, RMSprop
from sklearn.utils import class_weight
from utils import generator_batch_triplet, generator_batch
from keras.utils.training_utils import multi_gpu_model
from loss import triplet_loss, identity_loss, MARGIN

np.random.seed(1024)

FINE_TUNE = True
SAVE_FILTERED_LIST = True
FINE_TUNE_ON_ATTRIBUTES = True
LEARNING_RATE = 0.00001
NBR_EPOCHS = 100
BATCH_SIZE = 8
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'loss'
NBR_MODELS = 250
NBR_COLORS = 7
RANDOM_SCALE = True
nbr_gpus = len(GPUS.split(','))
INITIAL_EPOCH = 0

train_path = './train_vehicleModelColor_list.txt'
val_path = './val_vehicleModelColor_list.txt'

def filter_data_list(data_list):
    # data_list  : a list of [img_path, vehicleID, modelID, colorID]
    # {modelID: {colorID: {vehicleID: [imageName, ...]}}, ...}
    # dic helps us to sample positive and negative samples for each anchor.
    # https://arxiv.org/abs/1708.02386
    # The original paper says that "only the hardest triplets in which the three images have exactly
    # the same coarse-level attributes (e.g. color and model), can be used for similarity learning."
    dic = { }
    # We construct a new data list so that we could sample enough positives and negatives.
    new_data_list = [ ]
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        dic.setdefault(modelID, { })
        dic[modelID].setdefault(colorID, { })
        dic[modelID][colorID].setdefault(vehicleID, [ ]).append(imgPath)

    # https://stackoverflow.com/questions/11277432/how-to-remove-a-key-from-a-python-dictionary
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        #print(imgPath, vehicleID, modelID, colorID)
        if modelID in dic and colorID in dic[modelID] and vehicleID in dic[modelID][colorID] and \
                                                      len(dic[modelID][colorID][vehicleID]) == 1:
            dic[modelID][colorID].pop(vehicleID, None)

    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        if modelID in dic and colorID in dic[modelID] and len(dic[modelID][colorID].keys()) == 1:
            dic[modelID].pop(colorID, None)

    for modelID in dic:
        for colorID in dic[modelID]:
            for vehicleID in dic[modelID][colorID]:
                for imgPath in dic[modelID][colorID][vehicleID]:
                    new_data_list.append('{} {} {} {}'.format(imgPath, vehicleID, modelID, colorID))

    print('The original data list has {} samples, the new data list has {} samples.'.format(
                                 len(data_list), len(new_data_list)))
    return new_data_list, dic

if __name__ == "__main__":

    if FINE_TUNE:
        model_path = './models/triplet_models_backup/InceptionV3_Triplet_epoch=0026-loss=0.9686-modelAcc=0.9909-colorAcc=0.9617-val_loss=1.2408-val_modelAcc=0.9899-val_colorAcc=0.9350.h5'
        print('Finetune and Loading {} ...'.format(model_path))
        model = load_model(model_path, custom_objects={'identity_loss': identity_loss, 'triplet_loss': triplet_loss, 'MARGIN': MARGIN})
        INITIAL_EPOCH = 26

    elif FINE_TUNE_ON_ATTRIBUTES:
        print('Finetune on the attributes model ...')
        # Begin with Attributes pretrained weights.
        attributes_branch = load_model('./models/attributes_models/InceptionV3_vehicleModelColor_facs=1024_epoch=0004-loss=0.8892-modelAcc=0.8590-colorAcc=0.8990-val_loss=0.7493-val_modelAcc=0.8858-val_colorAcc=0.8890.h5')
        #attributes_branch.summary()
        attributes_branch.get_layer(name = 'global_average_pooling2d_1').name = 'f_base'
        f_base = attributes_branch.get_layer(name = 'f_base').output  # 1024-D

        anchor = attributes_branch.input
        positive = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='positive')
        negative = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='negative')

        # Attributes Branch
        f_acs = attributes_branch.get_layer(name = 'f_acs').output
        f_model = attributes_branch.get_layer(name = 'predictions_model').output
        f_color = attributes_branch.get_layer(name = 'predictions_color').output

        # Similarity Learning Branch
        f_sls1 = Dense(1024, name = 'sls1')(f_base)
        f_sls2 = concatenate([f_sls1, f_acs], axis = -1, name = 'sls1_concatenate')  # 1024-D
        # The author said that only layer ``SLS_2'' is applied ReLU since non-linearity
        # would disrupt the embedding learned in the layer ``SLS_1''.
        #f_sls2 = Activation('relu', name = 'sls1_concatenate_relu')(f_sls2)
        f_sls2 = Dense(1024, name = 'sls2')(f_sls2)
        f_sls2 = Activation('relu', name = 'sls2_relu')(f_sls2)
        # Non-linearity ?
        f_sls3 = Dense(256, name = 'sls3')(f_sls2)
        sls_branch = Model(inputs = attributes_branch.input, outputs = f_sls3)
        f_sls3_anchor = sls_branch(anchor)
        f_sls3_positive = sls_branch(positive)
        f_sls3_negative = sls_branch(negative)

        loss = Lambda(triplet_loss,
                  output_shape=(1, ))([f_sls3_anchor, f_sls3_positive, f_sls3_negative])

        model = Model(inputs = [anchor, positive, negative], outputs = [f_model, f_color, loss])
    else:
        # Begin with Imagenet pretrained weights.
        print('Loading InceptionV3 Weights from ImageNet Pretrained ...')
        inception = InceptionV3(include_top=False, weights= None,
               input_tensor=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling = 'avg')
        f_base = inception.get_layer(index = -1).output     # shape=(None, 1, 1, 2048)

        anchor = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='anchor')
        positive = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='positive')
        negative = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='negative')
        # Attributes-Model-Color Branch
        f_acs = Dense(1024, name='f_acs')(f_base)
        attributes_branch = Model(inputs= inception.input, outputs = f_acs)
        f_anchor_acs = attributes_branch(anchor)
        f_model = Dense(NBR_MODELS, activation='softmax', name='predictions_model')(f_anchor_acs)
        f_color = Dense(NBR_COLORS, activation='softmax', name='predictions_color')(f_anchor_acs)

        # Similarity Learning Branch
        f_sls1 = Dense(1024, name = 'sls1')(f_base)
        f_sls2 = concatenate([f_sls1, f_acs], axis = -1)
        # Non-linearity ?
        f_sls2 = Dense(1024, name = 'sls2')(f_sls2)
        # Non-linearity ?
        f_sls3 = Dense(256, name = 'sls3')(f_sls2)
        sls_branch = Model(inputs = inception.input, outputs = f_sls3)
        f_sls3_anchor = sls_branch(anchor)
        f_sls3_positive = sls_branch(positive)
        f_sls3_negative = sls_branch(negative)

        loss = Lambda(triplet_loss,
                  output_shape=(1, ))([f_sls3_anchor, f_sls3_positive, f_sls3_negative])

        model = Model(inputs = [anchor, positive, negative], outputs = [f_model, f_color, loss])

    print('Training model begins...')

    optimizer = SGD(lr = LEARNING_RATE, momentum = 0.9, decay = 0.0, nesterov = True)
    #optimizer = RMSprop(lr = LEARNING_RATE)

    if nbr_gpus > 1:
        print('Using {} GPUS.\n'.format(nbr_gpus))
        model = multi_gpu_model(model, gpus = nbr_gpus)
        BATCH_SIZE *= nbr_gpus
    else:
        print('Using a single GPU.\n')
    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy", identity_loss],
                  optimizer=optimizer, metrics=["accuracy"])

    #model.summary()

    model_file_saved = "./models/triplet_models_continue/InceptionV3_Triplet_epoch={epoch:04d}-loss={loss:.4f}-modelAcc={predictions_model_acc:.4f}-colorAcc={predictions_color_acc:.4f}-val_loss={val_loss:.4f}-val_modelAcc={val_predictions_model_acc:.4f}-val_colorAcc={val_predictions_color_acc:.4f}.h5"
    # Define several callbacks

    checkpoint = ModelCheckpoint(model_file_saved, verbose = 1)

    reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
                  patience=5, verbose=1, min_lr=0.00001)

    early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=15, verbose=1)

    train_data_lines = open(train_path).readlines()
    # Check if image path exists.
    train_data_lines = [w for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    train_data_lines, dic_train_data_lines = filter_data_list(train_data_lines)
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train * 1. / BATCH_SIZE))

    if SAVE_FILTERED_LIST:
        # Write filtered data lines into disk.
        filtered_train_list_path = './train_vehicleModelColor_list_filtered.txt'
        f_new_train_list = open(filtered_train_list_path, 'w')
        for line in train_data_lines:
            f_new_train_list.write(line + '\n')
        f_new_train_list.close()
        print('{} has been successfully saved!'.format(filtered_train_list_path))

    val_data_lines = open(val_path).readlines()
    val_data_lines = [w for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(nbr_val))
    validation_steps = int(ceil(nbr_val * 1. / BATCH_SIZE))

    model.fit_generator(generator_batch_triplet(train_data_lines, dic_train_data_lines,
                        mode = 'train', nbr_class_one = NBR_MODELS, nbr_class_two = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                        shuffle = True, augment = True),
                        steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = generator_batch_triplet(val_data_lines, { },
                        mode = 'val', nbr_class_one = NBR_MODELS, nbr_class_two = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps = validation_steps,
                        callbacks = [checkpoint], initial_epoch = INITIAL_EPOCH,
                        max_queue_size = 100, workers = 10, use_multiprocessing=True)
