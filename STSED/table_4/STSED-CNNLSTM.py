# LSTM-SED
import tensorflow as tf
from tensorflow.python.keras.layers import merge
from tensorflow.python.keras.layers.merge import Multiply
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.models import *

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Embedding, LSTM, Bidirectional, Permute
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import pandas as pd
from pandas import concat, read_csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sklearn.metrics as skm
import math
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from math import sqrt
import time

# fix random seed for reproducibility
np.random.seed(7)

start = time.time()

# 1. load dataset
dataframe = read_csv('../../XI-2/data/try2ed/PD_DJH.csv', engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
data = dataset.reshape(-1, 6)

timestep = 6
dim = 2

# 数据缩放 拆分输入X（7维）&输出Y（1维）
X = dataset[:, 0:2]
scaler = MinMaxScaler(feature_range=(0, 1))
for i in range(dim):
    Xdata = X[:, i]
    Xdata = Xdata.reshape(-1, 1)
    Xdata = scaler.fit_transform(Xdata)
    Xdata = Xdata.flatten()
    X[:, i] = Xdata
X_scaler = X.reshape(324, -1)
print(X_scaler.shape)

Y = dataset[:, 0]
Y_scaler = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
Y_scaler = Y_scaler.reshape(-1)
print(Y_scaler.shape)


# 重构数据集
##timestep为时间步长
def create_X(seq, timestep):
    dataX = []
    for i in range(len(seq) - timestep):
        a = seq[i:(i + timestep)]
        # X按照顺序取值 每次在后面增加一个数据
        dataX.append(a)
    return np.array(dataX)


def create_Y(seq, timestep):
    dataY = []
    for i in range(len(seq) - timestep):
        # Y向后移动一位取值
        dataY.append(seq[i + timestep])
    return np.array(dataY)


def get_lstm_model():
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(timestep, dim,))
    lstm_units1 = 12
    lstm_units2 = 36
    lstm_out1 = LSTM(lstm_units1, return_sequences=True)(inputs)
    lstm_out2 = LSTM(lstm_units2, return_sequences=True)(lstm_out1)
    dropout_out = Dropout(0.5)(lstm_out2)
    lstm_out = GlobalAveragePooling1D()(dropout_out)
    output = Dense(1, activation='relu')(lstm_out)
    model = Model(inputs=inputs, outputs=output)
    return model


# 预测
pres = []
# 将数据拆分成训练和测试，7/9作为训练数据
train_size = int(len(dataset) * 0.65)
test_size = len(dataset) - train_size
trainX, testX = X_scaler[0:train_size], X_scaler[train_size:len(dataset)]
trainY, testY = Y_scaler[0:train_size], Y_scaler[train_size:len(dataset)]
print("原始训练集的长度：", train_size)
print("原始测试集的长度：", test_size)
# print(trainX,trainX.shape)
# print(trainY,trainY.shape)
# print(testX,testX.shape)
# print(testY,testY.shape)

train_X = create_X(trainX, timestep)  # (246,6)
# print(train_X,train_X.shape)
train_Y = create_Y(trainY, timestep)  # (246,)
# print(train_Y,train_Y.shape)
test_X = create_X(testX, timestep)  # (66,6,6)
# print(test_X,test_X.shape)
test_Y = create_Y(testY, timestep)  # (66,)
# print(test_Y,test_Y.shape)

if __name__ == '__main__':

    model = get_lstm_model()
    optimizer = Adam(0.01)
    model.compile(optimizer=optimizer, loss='mse')
    print(model.summary())

    model.fit(train_X, train_Y, epochs=700, batch_size=64)

    # 开始预测
    trainPredict = model.predict(train_X)
    testPredict = model.predict(test_X)

    # 逆缩放预测值
    dataframe = read_csv('../../XI-2/data/try2ed/PD.csv', engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    Y = dataset[:, 0]
    # trainPredict = trainPredict*((numpy.max(Y)-numpy.min(Y)))+numpy.min(Y)
    # trainY_ori = trainY*((numpy.max(Y)-numpy.min(Y)))+numpy.min(Y)
    # trainY_ori=trainY_re.reshape((246,1))
    testPre = testPredict * ((np.max(Y) - np.min(Y))) + np.min(Y)
    testPre = testPre.reshape(-1)
    Y = dataset[:, 0]
    Y = Y[train_size + timestep:len(dataset)]
    # testY_ori = testY*((np.max(dataset)-np.min(dataset)))+np.min(dataset)
    # testY_ori = testY_ori.reshape((-1,1))
    testY_ori = Y.reshape(-1)

    # 计算误差
    # trainScore = math.sqrt(mean_squared_error(trainY_ori[:,0], trainPredict[:,0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(testY_ori[:,0], testPredict[:,0]))
    # print('Test Score: %.2f RMSE' % (testScore))

    error = []  # Y-Y'
    error1 = []  # abs((Y-Y')/Y)
    error2 = []  # Y*Y'
    squared1 = []  # Y*Y
    squared2 = []  # Y'*Y'
    for i in range(len(testY_ori)):
        error.append(testY_ori[i] - testPre[i])
        error1.append(abs((testY_ori[i] - testPre[i]) / testY_ori[i]))
        error2.append(testY_ori[i] * testPre[i])
        squared1.append(testY_ori[i] * testY_ori[i])
        squared2.append(testPre[i] * testPre[i])

    squaredError = []  # (Y-Y')^2
    absError = []  # abs(Y-Y')
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值

    MSE = sum(squaredError) / len(squaredError)
    meannn = np.mean(testY_ori)
    NMSEerror = []
    IAerror = []
    for i in range(len(testY_ori)):
        NMSEerror.append(squaredError[i] / error2[i])
        IAerror.append((abs(testY_ori[i] - meannn) + abs(testPre[i] - meannn)) * (
                    abs(testY_ori[i] - meannn) + abs(testPre[i] - meannn)))

    U2error1 = []
    U2error2 = []
    for i in range(len(testY_ori) - 1):
        U2error1.append(
            ((testY_ori[i + 1] - testPre[i + 1]) / testY_ori[i]) * ((testY_ori[i + 1] - testPre[i + 1]) / testY_ori[i]))
        U2error2.append(
            ((testY_ori[i + 1] - testPre[i]) / testY_ori[i]) * ((testY_ori[i + 1] - testPre[i]) / testY_ori[i]))
    print('\n==========================')
    MAE = sum(absError) / len(absError)
    print('MAE=', MAE)
    from math import sqrt

    RMSE = sqrt(MSE)
    print("RMSE = ", RMSE)  # 均方根误差RMSE
    NMSE = sum(NMSEerror)
    print("NMSE = ", NMSE)  # 误差平方的归一化平均值NMSE
    MAPE = sum(error1) / len(error1)
    print('MAPE=', MAPE)
    IA = 1 - sum(squaredError) / sum(IAerror)
    print('IA=', IA)  # 一致性指数
    U1index = RMSE / (sqrt(sum(squared1) / len(squared1)) + sqrt(sum(squared2) / len(squared2)))
    print('U1=', U1index)
    U2index = (sqrt(sum(U2error1) / len(testY_ori))) / (sqrt(sum(U2error2) / len(testY_ori)))
    print('U2=', U2index)

    # print(testPredict)
    testPre = testPre.reshape(-1)
    testPre = pd.DataFrame(testPre)
    testPre.to_csv('../XI-2/data/pre_result/lstmoutput.csv', header=False)
    import csv

    Evaluation_index = [MAE, RMSE, NMSE, MAPE, IA, U1index, U2index]
    with open("../../XI-2/data/pre_result/lstmEvaluation.csv", "a", newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(Evaluation_index)

    plt.figure(facecolor='white')
    plt.plot(testPre, color='green', label='Predict')
    plt.plot(testY_ori, color='pink', label='Original')
    plt.legend(loc='best')
    plt.show()
    pres.append(testPre)

end = time.time()
print('Running time: %d seconds' % (end - start))