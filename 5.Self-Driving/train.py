import numpy as np
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import  Conv2D, MaxPooling2D, Flatten, PReLU
from keras.models import Sequential, Model
from keras import backend as K
from keras.regularizers import l2
import csv
import cv2
import glob
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from keras import callbacks
import math
from matplotlib import pyplot
import tensorboard

SEED = 13

def horizontal_flip(img, degree):
    '''
    按照50%的概率水平翻转图像
    img: 输入图像
    degree: 输入图像对于的转动角度
    '''

    return img, degree


def random_brightness(img, degree):
    '''
    随机调整输入图像的亮度， 调整强度于 0.1(变黑)和1(无变化)之间
    img: 输入图像
    degree: 输入图像对于的转动角度
    '''

    return img, degree


def left_right_random_swap(img_address, degree, degree_corr=1.0 / 4):
    '''
    随机从左， 中， 右图像中选择一张图像， 并相应调整转动的角度
    img_address: 中间图像的的文件路径
    degree: 中间图像对于的方向盘转动角度
    degree_corr: 方向盘转动角度调整的值
    '''
    return img_address, degree


def discard_zero_steering(degrees, rate):
    '''
    从角度为零的index中随机选择部分index返回
    degrees: 输入的角度值
    rate: 丢弃率， 如果rate=0.8， 意味着80%的index会被返回， 用于丢弃
    '''

    ans = np.where(np.array(degrees) == 0)
    # print(np.array(ans).size)
    return ans[0: int(np.array(ans).size * rate)]


def get_model(shape):
    '''
    预测方向盘角度: 以图像为输入, 预测方向盘的转动角度
    shape: 输入图像的尺寸, 例如(128, 128, 3)
    '''
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu', input_shape=shape))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu'))

    model.add(Flatten())

    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error')

    return model


def image_transformation(img_address, degree, data_dir):
    #img_address, degree = left_right_random_swap(img_address, degree)
    img = cv2.imread(data_dir + img_address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img, degree = random_brightness(img, degree)
    #img, degree = horizontal_flip(img, degree)

    return img, degree


def batch_generator(x, y, batch_size, shape, training=True, data_dir='data/', monitor=True, yieldXY=True,
                    discard_rate=0.95):
    """
    产生批处理的数据的generator
    x: 文件路径list
    y: 方向盘的角度
    training: 值为True时产生训练数据
              值为False时产生validation数据
    batch_size: 批处理大小
    shape: 输入图像的尺寸(高, 宽, 通道)
    data_dir: 数据目录, 包含一个IMG文件夹
    monitor: 保存一个batch的样本为 'X_batch_sample.npy‘ 和'y_bag.npy’
    yieldXY: 为True时, 返回(X, Y)
             为False时, 只返回 X only
    discard_rate: 随机丢弃角度为零的训练数据的概率
    """

    if training:
        y_bag = []
        # 洗牌，每次调用这个函数，都会出现新的排序，减少相关性的影响
        x, y = shuffle(x, y)
        rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
        new_x = np.delete(x, rand_zero_idx, axis=0)
        new_y = np.delete(y, rand_zero_idx, axis=0)
    else:
        new_x = x
        new_y = y

    offset = 0
    while True:
        X = np.empty((batch_size, *shape))
        Y = np.empty((batch_size, 1))

        for example in range(batch_size):
            img_address, img_steering = new_x[example + offset], new_y[example + offset]

            if training:
                img, img_steering = image_transformation(img_address, img_steering, data_dir)
            else:
                img = cv2.imread(data_dir + img_address)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            X[example, :, :, :] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1])) / 255 - 0.5

            Y[example] = img_steering
            if training:
                y_bag.append(img_steering)

            '''
             到达原来数据的结尾, 从头开始
            '''
            if (example + 1) + offset > len(new_y) - 1:
                x, y = shuffle(x, y)
                rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
                new_x = x
                new_y = y
                new_x = np.delete(new_x, rand_zero_idx, axis=0)
                new_y = np.delete(new_y, rand_zero_idx, axis=0)
                offset = 0
        if yieldXY:
            yield (X, Y)
        else:
            yield X

        offset = offset + batch_size
        if training:
            np.save('y_bag.npy', np.array(y_bag))
            np.save('Xbatch_sample.npy', X)


if __name__ == '__main__':

    data_path = 'data/'
    with open(data_path + 'driving_log.csv', 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        log = []
        for row in file_reader:
            log.append(row)

    log = np.array(log)
    # 去掉文件第一行
    log = log[1:, :]

    # 判断图像文件数量是否等于csv日志文件中记录的数量
    ls_imgs = glob.glob(data_path + 'IMG/*.jpg')
    assert len(ls_imgs) == len(log) * 3, 'number of images does not match'

    # 使用20%的数据作为测试数据
    validation_ratio = 0.2
    shape = (128, 128, 3)
    batch_size = 32
    nb_epoch = 10

    x_ = log[:, 0]
    y_ = log[:, 3].astype(float)
    x_, y_ = shuffle(x_, y_)
    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=validation_ratio, random_state=SEED)

    print('batch size: {}'.format(batch_size))
    print('Train set size: {} | Validation set size: {}'.format(len(X_train), len(X_val)))

    samples_per_epoch = batch_size
    # 使得validation数据量大小为batch_size的整数陪
    nb_val_samples = len(y_val) - len(y_val) % batch_size
    model = get_model(shape)
    print(model.summary())

    # 根据validation loss保存最优模型
    save_best = callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='min')

    # 如果训练持续没有validation loss的提升, 提前结束训练
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15,
                                         verbose=0, mode='auto')
    tbCallback = callbacks.TensorBoard(log_dir='./Graph', write_graph=True)
    callbacks_list = [early_stop, save_best, tbCallback]

    history = model.fit_generator(batch_generator(X_train, y_train, batch_size, shape, training=True),
                                  steps_per_epoch=samples_per_epoch,
                                  validation_steps=nb_val_samples // batch_size,
                                  validation_data=batch_generator(X_val, y_val, batch_size, shape,
                                                                  training=False, monitor=False),
                                  epochs=nb_epoch, verbose=1, callbacks=callbacks_list)

    with open('./trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig('train_val_loss.png')

    # 保存模型
    with open('model.json', 'w') as f:
        f.write(model.to_json())
    model.save('model.h5')
    print('Done!')


















