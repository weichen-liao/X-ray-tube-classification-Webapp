# -*- coding: utf-8 -*-
# Author: Weichen Liao

# -*- coding: utf-8 -*-
# Author: Weichen Liao

import os
import random
import glob
from shutil import copyfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB7, InceptionV3, MobileNet
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from PIL import Image, ImageFont, ImageDraw
import seaborn as sns
import matplotlib.pyplot as plt

TRAIN_PATH = './ranzcr-clip-catheter-line-classification/train'
TEST_PATH = './ranzcr-clip-catheter-line-classification/test'
TRAIN_SMALL_PATH = './train-small'
VALIDATION_SMALL_PATH = './validation-small'
TEST_SMALL_PATH = './test-small'
LABEL_FILE = pd.read_csv('./ranzcr-clip-catheter-line-classification/train.csv')
LABELS = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged',
          'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']
MODEL_PATH = 'models/best_model.hdf5'
FONT = './font/arial.ttf'
IMAGE_LENGTH, IMAGE_HEIGHT, NUM_CHANNEL = 224, 224, 3
COLOR_GREEN = (40, 250, 40, 150)


class Config:
    batch_size = 32
    n_epochs = 15
    # optimizer
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    # criterion
    loss = 'binary_crossentropy'
    metrics = tf.keras.metrics.AUC(multi_label=True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=1, mode='auto',
                                       epsilon=0.0001,
                                       cooldown=5, min_lr=0.00001)

    checkpoint = ModelCheckpoint('models/best_model.hdf5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=False)

    checkpoint_last = ModelCheckpoint('models/last_model.hdf5',
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=False,
                                      mode='min',
                                      save_weights_only=False)

    early = EarlyStopping(monitor='val_loss',
                          mode='min',
                          patience=5)

CONFIG = Config

# remove everything is a directory
def ClearFolder(dir):
    files = glob.glob(dir)
    for f in files:
        if f.endswith('.jpg'):
            os.remove(f)

# given a StudyInstanceUID, return a dictionary of its labels
def CheckLabels(StudyInstanceUID: str):
    dict_labels = {}
    cell = LABEL_FILE[LABEL_FILE['StudyInstanceUID'] == StudyInstanceUID]
    for l in LABELS:
        dict_labels[l] = cell[l].values[0]
    return dict_labels

# take a small sample from train and test, put them into different folders according to their labels, each label has a folder
def SampleImages(num_train: int, num_valid: int, num_test: int):
    # remove all files in the target directory
    print('Clearing the folders...')
    ClearFolder(TRAIN_SMALL_PATH+'/*')
    for l in tqdm(LABELS):
        ClearFolder(TRAIN_SMALL_PATH+'/'+l+'/*')
    for l in tqdm(LABELS):
        ClearFolder(VALIDATION_SMALL_PATH+'/'+l+'/*')
    ClearFolder(TEST_SMALL_PATH+'/*')

    # take samples from train and test
    train_all = [f for f in os.listdir(TRAIN_PATH) if os.path.isfile(os.path.join(TRAIN_PATH, f))]
    train_sampled = random.sample(train_all, num_train)
    validation_sampled = random.sample(train_all, num_valid)
    test_all = [f for f in os.listdir(TEST_PATH) if os.path.isfile(os.path.join(TEST_PATH, f))]
    test_sampled = random.sample(test_all, num_test)

    # copy the sampled files to the target folder according to its label
    print('Copying the images...')
    for f in tqdm(train_sampled):
        dict_labels = CheckLabels(f.rstrip('.jpg'))
        for l in LABELS:
            if dict_labels[l] == 1:
                copyfile(TRAIN_PATH+'/'+f, TRAIN_SMALL_PATH+'/'+l+'/'+f)
    for f in tqdm(validation_sampled):
        dict_labels = CheckLabels(f.rstrip('.jpg'))
        for l in LABELS:
            if dict_labels[l] == 1:
                copyfile(TRAIN_PATH+'/'+f, VALIDATION_SMALL_PATH+'/'+l+'/'+f)
    for f in test_sampled:
        copyfile(TEST_PATH+'/'+f, TEST_SMALL_PATH+'/'+f)


def AddTextToImage(image, df_pred: pd.DataFrame, size:int, color: tuple):
    '''
    :param image:
    :param df_pred: the dataframe of prediction probabilities, columns: probability
    :param size:
    :param color: example: (255, 255, 255, 128)
    :return:
    '''
    image = image.convert("RGBA")
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new("RGBA", image.size, (255, 255, 255, 0))
    # get a font
    fnt = ImageFont.truetype(FONT, size)
    # get a drawing context
    d = ImageDraw.Draw(txt)
    '''
    # decide which labels to put on the image and draw the text
    df_pred = df_pred.round(2)
    df_show = df_pred[df_pred['probability']>=0.7]
    if len(df_show)>0:
        for index, value in df_show['probability'].items():
            d.text((10, 10), str(index) + ':' + str(value), font=fnt, fill=color)
    else:
        df_show = df_pred.iloc[0]
        d.text((10, 10), df_show.name + ':' + str(df_show.values[0]), font=fnt, fill=color)
    '''

    # create a new column: the avergae prob of the other columns
    col = df_pred.loc[:, ['InceptionV3', 'MobileNet', 'EfficientNetB3', 'EfficientNetB7']]
    df_pred['avg_prob'] = col.mean(axis=1)
    # get the largest average probability
    series = df_pred['avg_prob']
    max_index = str(series.idxmax())
    max_value = str(series[max_index])
    # show it
    d.text((10, 10), max_index + ':' + max_value, font=fnt, fill=color)
    out = Image.alpha_composite(image, txt)
    return out


def PredictTestImages(model, save_path: str):
    '''
    :param df_test: dataframe of test, must contain the column: StudyInstanceUID
    :return:
    '''
    if save_path == '':
        raise Exception('Please specify the save path of final results')
    print('------------------------ Predicting the test images ------------------------')
    res = []
    files = glob.glob(TEST_PATH+'/*')
    for f in tqdm(files):
        if f.endswith('.jpg'):
            StudyInstanceUID = f.split('/')[-1].rstrip('.jpg')
            line = [StudyInstanceUID]
            probs = PredictSingleImage(image_path=f, model=model)
            line.extend(probs)
            res.append(line)
    columns = ['StudyInstanceUID']
    columns.extend(LABELS)
    res = pd.DataFrame(data=res, columns=columns)
    res.to_csv(save_path,index=False)
    return res


# the evaluation on the validation set will be decided based on test_test and random_state during the training process
def Evaluation(test_size: float, valid_size: float, random_state: int, model, save_path='', model_name='', rescale = False, resize_length=IMAGE_LENGTH, resize_height=IMAGE_HEIGHT):
    '''
    :param test_size:
    :param random_state:
    :param model:
    :param save_path: save the prediction(numpy array)
    :param model_name: str, the name of the model
    :return:
    '''
    def process_ds(x, y):
        file_path = TRAIN_PATH + '/' + x + '.jpg'
        x = tf.io.read_file(file_path)
        x = tf.image.decode_jpeg(x, channels=NUM_CHANNEL)
        if rescale == True:
            x = x/255
        x = tf.image.resize(x, [resize_length, resize_height])
        return x, y

    df_train, df_test = train_test_split(LABEL_FILE, test_size=test_size, random_state=random_state, shuffle=True)
    df_train, df_valid = train_test_split(df_train, test_size=valid_size, random_state=random_state, shuffle=True)
    print('length of train:', len(df_train))
    print('length of valid:', len(df_valid))
    print('length of test:', len(df_test))
    valid_labels = df_valid[LABELS].values
    valid_ds = tf.data.Dataset.from_tensor_slices((df_valid['StudyInstanceUID'].values, valid_labels))
    valid_ds = valid_ds.map(process_ds)
    valid_batch = valid_ds.batch(CONFIG.batch_size * 2)
    pred_valid_y = model.predict(valid_batch, verbose=True)
    pred_valid_y = np.array(pred_valid_y)
    if save_path != '':
        np.save(save_path, pred_valid_y)

    avg_auc = 0
    aucs = []
    for i, item in enumerate(LABELS):
        auc = roc_auc_score(valid_labels[:, i], pred_valid_y[:, i])
        aucs.append([auc])

        avg_auc += auc
    aucs = pd.DataFrame(aucs, columns=[model_name], index=LABELS)
    avg_auc = avg_auc/len(LABELS)
    return aucs, avg_auc


def RocCurve(test_size: float, valid_size: float, random_state: int, model_names: str, arrays: list, label_chosen: str):
    '''
    :param test_size:
    :param valid_size:
    :param random_state:
    :param model_names:
    :param arrs: the list of npy paths(the npy array of test probability)
    :param label_chosen: the label to plot
    :return:
    '''
    df_train, df_test = train_test_split(LABEL_FILE, test_size=test_size, random_state=random_state, shuffle=True)
    df_train, df_valid = train_test_split(df_train, test_size=valid_size, random_state=random_state, shuffle=True)
    if len(model_names) != len(arrays):
        raise Exception('the model_names and arrays not in the same length')
    fig, ax = plt.subplots(1, figsize=(7, 7))
    for i in range(len(model_names)):
        model_name = model_names[i]
        arr = np.load(arrays[i])
        y_test = df_valid[label_chosen].values
        y_prob = arr[:,LABELS.index(label_chosen)]
        fp, tp, threshold = roc_curve(y_test, y_prob)
        plt.plot(fp, tp, linewidth=1, label=model_name)

    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.legend()
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    return fig, ax

# given a image path, make predictions
def PredictSingleImage(image_path: str, model, rescale=False):
    '''
    :param image_path:
    :return: list / numpy array
    '''
    test_img = tf.io.read_file(image_path)
    test_img = tf.image.decode_jpeg(test_img, channels=NUM_CHANNEL)
    if rescale == True:
        test_img = test_img/255
    test_img = tf.image.resize(test_img, [IMAGE_LENGTH, IMAGE_HEIGHT])

    res = model.predict(tf.reshape(test_img, (1, IMAGE_LENGTH, IMAGE_HEIGHT, NUM_CHANNEL)))
    return res[0]

# The single image prediction function for Webapp
def PredictForWebApp(image, model, model_name: str, rescale=False):
    '''
    :param image: PIL.JpegImagePlugin.JpegImageFile
    :param model:
    :param model_name: the name of the model
    :param rescale:
    :return:
    '''
    # must convert into 3 channels, otherwise it will be just 1 channel
    image = image.convert("RGB")
    # type numpy.ndarray, shape:(length, height, 3)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    print(image_array.shape)
    if rescale == True:
        image_array = image_array/255
    test_img = tf.image.resize(image_array, [IMAGE_LENGTH, IMAGE_HEIGHT])
    preds = model.predict(tf.reshape(test_img, (1, IMAGE_LENGTH, IMAGE_HEIGHT, NUM_CHANNEL)))
    preds = pd.DataFrame(data=preds, columns=LABELS)
    preds = preds.round(2).T
    preds.columns = [model_name]
    preds = preds.sort_values(by=[model_name], ascending=False)
    return preds


def InitializeLoadedModel(model_path: str, model_chosen: str):
    if model_chosen == 'EfficientNetB3':
        model = EfficientNetB3(include_top=False, weights='imagenet', pooling='max')
    elif model_chosen == 'EfficientNetB7':
        model = EfficientNetB7(include_top=False, weights='imagenet', pooling='max')
    elif model_chosen == 'InceptionV3':
        model = InceptionV3(include_top=False, weights='imagenet', pooling='max')
    elif model_chosen == 'MobileNet':
        model = MobileNet(include_top=False, weights='imagenet', pooling='max')
    else:
        raise Exception('Invalid model chosen, it has to be [EfficientNetB3, EfficientNetB7, InceptionV3, MobileNet]')
    model.trainable = False

    inputs = tf.keras.layers.Input(shape=(IMAGE_LENGTH, IMAGE_HEIGHT, NUM_CHANNEL))

    outputs_eff = model(inputs)
    dense_1 = Dense(256)(outputs_eff)
    bn_1 = BatchNormalization()(dense_1)
    activation = Activation('relu')(bn_1)
    dropout = Dropout(0.3)(activation)
    dense_2 = Dense(len(LABELS), activation='sigmoid')(dropout)

    my_model = tf.keras.Model(inputs, dense_2)

    my_model.compile(
        optimizer=CONFIG.optimizer,
        loss=CONFIG.loss,
        metrics=CONFIG.metrics
    )
    my_model.load_weights(model_path)
    return my_model

# input a probability matrix with shape of (x,11), output and numpy matrix of same size, with predicted labels
#
def ProbToPred(matrix_prob: np.array, method: str, threhold=0.75):
    if method == 'max':
        res = np.zeros_like(matrix_prob)
        res[np.arange(len(matrix_prob)), matrix_prob.argmax(1)] = 1
        return res
    elif method == 'threhold':
        res = np.copy(matrix_prob)
        res[res >= threhold] = 1
        res[res < threhold] = 0
        return res
    else:
        raise Exception('unknow method, method must be max or threhold')


if __name__ == '__main__':
    # SampleImages(1000,1000,100)

    # model2 = InitializeLoadedModel(model_path='models/InceptionV3224_best_model.hdf5', model_chosen='InceptionV3')

    # df_auc2, avg_auc2 = Evaluation(test_size=0.1, valid_size= 0.08,random_state=1, model=model2, save_path = 'predicts/InceptionV3224_best_model.npy', model_name='InceptionV3224', rescale=True)
    # print(df_auc2)
    # print(avg_auc2)

    RocCurve(test_size=0.1, valid_size= 0.08,random_state=1, model_names=['InceptionV3', 'MobileNet', 'EfficientNetB3', 'EfficientNetB7'], arrays=['predicts/InceptionV3224_best_model.npy', 'predicts/MobileNet224_best_model.npy', 'predicts/EfficientNetB3224_best_model.npy', 'predicts/EfficientNetB7224_best_model.npy'])

    # pred = PredictSingleImage('test-small/1.2.826.0.1.3680043.8.498.10068042846486951897146262305314065898.jpg',  model=model)
    # print(pred)

    # PredictTestImages(model=model, save_path='submissions/efficientNet_0224.csv')

    # list_pred = Evualation(test_size=0.2, random_state=1, model=model)
    # list_pred = np.array(list_pred)
    # np.save('pred.npy', list_pred)

    # list_prob = np.load('predicts/prob_valid.npy')
    # print(list_prob.shape)
    # list_pred = ProbToPred(list_prob, method='threhold', threhold=0.75)
    # print(list_pred)
    # df_train, df_valid = train_test_split(LABEL_FILE, test_size=0.2, random_state=1, shuffle=True)
    # valid_labels = df_valid[LABELS].values
    # avg_auc = 0
    # for i, item in enumerate(LABELS):
    #     auc = roc_auc_score(valid_labels[:,i], list_prob[:, i])
    #     acc = accuracy_score(valid_labels[:,i], list_pred[:, i])
    #     print(item, 'AUC:',auc, 'ACC:', acc)
    #     avg_auc += auc
    # print('Average AUC:', avg_auc/11)
