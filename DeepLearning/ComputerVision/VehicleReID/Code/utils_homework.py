from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import Sequence
import PIL
from PIL import Image
import os
import numpy as np
import random
from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import time

np.random.seed(1024)

def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw : centerw + halfw,
                 centerh - halfh : centerh + halfh, :]

    return cropped

def scale_byRatio(img_path, ratio=1.0, return_width=299, crop_method=center_crop):
    # Given an image path, return a scaled array
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    shorter = min(w, h)
    longer = max(w, h)
    img_cropped = crop_method(img, (shorter, shorter))
    img_resized = cv2.resize(img_cropped, (return_width, return_width), interpolation=cv2.INTER_CUBIC)
    img_rgb = img_resized
    img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]

    return img_rgb

# Data Agumentation: https://github.com/aleju/imgaug

"""
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
st = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        #iaa.Flipud(0.5), # vertically flip 50% of all images
        st(iaa.Crop(percent=(0, 0.15))), # crop images by 0-15% of their height/width
        #st(iaa.GaussianBlur((0, 2.0))), # blur images with a sigma between 0 and 3.0
        st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        st(iaa.Multiply((0.85, 1.15), per_channel=0.5)), # change brightness of images (75-125% of original value)
        st(iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)), # improve or worsen the contrast
        st(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            #translate_px={"x": (-10, 10), "y": (-10, 10)}, # translate by -16 to +16 pixels (per axis)
            rotate=(-15, 15), # rotate by -10 to +10 degrees
            #shear=(-5, 5), # shear by -16 to +16 degrees
            order=ia.ALL, # use any of scikit-image's interpolation methods
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ))
    ],
    random_order=True # do all of the above in random order
)
"""

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.85, 1.15), "y": (0.85, 1.5)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15), # rotate by -45 to +45 degrees
            shear=(-5, 5), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(1, 5)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(1, 5)), # blur image using local medians with kernel sizes between 2 and 7
                ]),

                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.01, 0.03), per_channel=0.2),
                ]),
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                iaa.ContrastNormalization((0.3, 1.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
            ],
            random_order=True
        )
    ],
    random_order=True
)

def generator_batch(data_list, nbr_classes=3, batch_size=32, return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):
    '''
    A generator that yields a batch of (data, label).

    Input:
        data_list  : a MxNet styple of data list, e.g.
                     "/data/workspace/dataset/Cervical_Cancer/train/Type_1/0.jpg 0"
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size

    Output:
        (X_batch, Y_batch)
    '''

    N = len(data_list)

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch = np.zeros((current_batch_size, nbr_classes))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print line
            if return_label:
                label = int(line[-1])
            img_path = line[0]

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)
            img = scale_byRatio(img_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            X_batch[i - current_index] = img
            if return_label:
                Y_batch[i - current_index, label] = 1

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)

        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path = data_list[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(save_to_dir, image_name)
                img = array_to_img(X_batch[i - current_index])
                img.save(img_to_save_path)

        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        if save_network_input:
            print('X_batch.shape: {}'.format(X_batch.shape))
            X_to_save = X_batch.reshape((299, 299, 3))
            to_save_base_name = save_network_input[:-4]
            np.savetxt(to_save_base_name + '_0.txt', X_to_save[:, :, 0], delimiter = ' ')
            np.savetxt(to_save_base_name + '_1.txt', X_to_save[:, :, 1], delimiter = ' ')
            np.savetxt(to_save_base_name + '_2.txt', X_to_save[:, :, 2], delimiter = ' ')

        img = X_batch[0,:,:,:]
        img = np.reshape(img, (-1))
        if return_label:
            yield (X_batch, Y_batch)
        else:
            yield X_batch

def generator_batch_multitask(data_list, nbr_class_one=250, nbr_class_two=7, batch_size=32, return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    preprocess=False, img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, augment=False):
    '''
    A generator that yields a batch of (data, class_one, class_two).

    Input:
        data_list  : a MxNet styple of data list, e.g.
                     "/data/workspace/dataset/Cervical_Cancer/train/Type_1/0.jpg 0"
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size

    Output:
        (X_batch, Y1_batch, Y2_batch)
    '''

 

def generator_batch_triplet(data_list, dic_data_list, nbr_class_one=250, nbr_class_two=7,
                    batch_size=32, return_label=True, mode='train',
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):
    '''
    A generator that yields a batch of ([anchor, positive, negative], [class_one, class_two, pseudo_label]).

    Input:
        data_list  : a list of [img_path, vehicleID, modelID, colorID]
        dic_data_list: a dictionary: {modelID: {colorID: {vehicleID: [imageName, ...]}}, ...}
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size
        mode       : generator used as for 'train', 'val'.
                     if mode is set 'train', dic_data_list has to be specified.
                     if mode is set 'val', dic_data_list could be a null dictionary: { }.
                     if mode is et 'feature_extraction', then return (X_anchor)


    Output:
        ([anchor, positive, negative], [class_one, class_two, pseudo_label]
    '''
 

def generator_batch_classware(data_list_ori, nbr_classes=3, batch_size=32, return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    preprocess=False, img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, augment=False):
    '''
    A generator that yields a batch of (data, label).
    Note that in each mini-batch, each data is sampled uniformly, i.e.
    data is balanced in each batch. It is a quite useful trick in the case of
    unbalanced data set.

    Input:
        data_list  : a MxNet styple of data list, e.g.
                     "/data/workspace/dataset/Cervical_Cancer/train/Type_1/0.jpg 0"
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size

    Output:
        (X_batch, Y_batch)
    '''
    data_list = [[] for i in range(nbr_classes)]
    for line in data_list_ori:
        label_temp = line.strip().split(' ')[-1]
        data_list[int(label_temp)].append(line)

    N = np.zeros(nbr_classes, int)
    for i in range(nbr_classes):
        N[i] = len(data_list[i])
    index = np.zeros(nbr_classes, int)

    num_per_label = batch_size // nbr_classes
    while True:
        X_batch = np.zeros((batch_size, img_width, img_height, 3))
        Y_batch = np.zeros((batch_size, nbr_classes))
        list_this_batch = []
        for i in range(nbr_classes):
            if index[i]+num_per_label>=N[i]-1:
                index[i] = 0
                if shuffle:
                    random.shuffle(data_list[i])
            for j in range(num_per_label):
                id = index[i]+j
                if id > N[i]:
                    id = id%N[i]
                list_this_batch.append(data_list[i][id])
            index[i] += num_per_label
        random.shuffle(list_this_batch)
        for i in range(batch_size):
            line = list_this_batch[i].strip().split(' ')
            #print line
            img_path = line[0]
            #img = load_img(img_path, target_size = (img_width, img_height))
            #img = img_to_array(img)
            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)
            img = scale_byRatio(img_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            X_batch[i] = img
            if return_label:
                label = int(line[-1])
                Y_batch[i, label] = 1

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)

        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path = data_list[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(save_to_dir, image_name)
                img = array_to_img(X_batch[i - current_index])
                img.save(img_to_save_path)

        if preprocess:
            for i in range(batch_size):
                X_batch[i] = preprocessing_eye(X_batch[i], return_image=True,
                                               result_size=(img_width, img_height))

        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        if return_label:
            yield (X_batch, Y_batch)
        else:
            yield X_batch


def test_generator_batch(data_list, batch_size=1, img_width=299, img_height=299,
                         scale_ratio=1.0, crop_method=center_crop,
                         dense_output = False, stride_x = 1, stride_y = 1,
                         scale_crop_biasRatio = False, biasRatio=0.5):
    N = len(data_list)
    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))

        for i in range(current_index, current_index + current_batch_size):
            img_path = data_list[i].strip()
            if dense_output:
                img = dense_byStride(img_path, ratio = scale_ratio, stride_x = stride_x,
                                     stride_y = stride_y, return_width = img_width)
            elif scale_crop_biasRatio:
                img = scale_byRatio_crop_biasRatio(img_path, biasRatio, img_width)
            else:
                img = scale_byRatio(img_path, ratio=scale_ratio,
                                   return_width=img_width, crop_method=crop_method)

            X_batch[i - current_index] = img

        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        yield X_batch