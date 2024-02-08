# ConvLSTM-SED   multivariable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# matplotlib inline
import time

# fix random seed for reproducibility
np.random.seed(7)

start = time.time()

# load the dataset
# dataframe = read_csv('C:/Users/Administrator/Desktop/XI-2/data/PD-TVFEMD-IMF1.csv', usecols=[6], engine='python')
dataframe = read_csv('../../XI-2/data/try2ed/PD-TVFEMD-BDX.csv', engine='python')
# print(dataframe)
print("数据集的长度：", len(dataframe))
dataset = dataframe.values
# 将整型变为float
dataset = dataset.astype('float32')
# print(dataset)

timestep = 6
dim = 6
# 数据缩放 拆分输入X（7维）&输出Y（1维）
X = dataset[:, 0:6]
scaler = MinMaxScaler(feature_range=(0, 1))
for i in range(dim):
    Xdata = X[:, i]
    Xdata = Xdata.reshape(-1, 1)
    Xdata = scaler.fit_transform(Xdata)
    Xdata = Xdata.flatten()
    X[:, i] = Xdata
# X_scaler= scaler.fit_transform(X)
X_scaler = X.reshape(324, -1)

Y = dataset[:, 0]
Y_scaler = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
Y_scaler = Y_scaler.reshape(-1)

# print(X_scaler)
# print(Y_scaler)
# print(Y)


# 将数据拆分成训练和测试，8/9作为训练数据
train_size = int(len(dataset) * 0.65)
test_size = len(dataset) - train_size
trainX, testX = X_scaler[0:train_size, :], X_scaler[train_size:len(dataset), :]
trainY, testY = Y_scaler[0:train_size], Y_scaler[train_size:len(dataset)]
print("原始训练集的长度：", train_size)
print("原始测试集的长度：", test_size)


# print(trainX)
# print(trainY)
# print(testX)
# print(testY)


# 重构数据集
##timestep为时间步长
def create_trainX(seq, timestep):
    dataX = []
    for i in range(len(seq) - timestep):
        a = seq[i:(i + timestep)]
        # X按照顺序取值 每次在后面增加一个数据
        dataX.append(a)
    return np.array(dataX)


def create_trainY(seq, timestep):
    dataY = []
    for i in range(len(seq) - timestep):
        # Y向后移动一位取值
        dataY.append(seq[i + timestep])
    return np.array(dataY)


def create_testX(seq, timestep):
    dataX = []
    for i in range(len(seq) - timestep):
        a = seq[(i):(i + timestep)]
        # X按照顺序取值 每次在后面增加一个数据
        dataX.append(a)
    return np.array(dataX)


# ----------forecasting-----------

pres = []
trainX = create_trainX(trainX, timestep)
trainY = create_trainY(trainY, timestep)
testX = create_testX(testX, timestep)
testY = testY[timestep:len(testY)]
print("转为监督学习，训练集数据长度：", len(trainX))
# print(trainX,trainY)
print("转为监督学习，测试集数据长度：", len(testX))
# print(testX, testY )


# 数据重构为5D [samples, timesteps, rows, columns, features]
trainX_input5D = np.reshape(trainX, (trainX.shape[0], 2, 1, timestep // 2, dim))
testX_input5D = np.reshape(testX, (testX.shape[0], 2, 1, timestep // 2, dim))
# trainX_input5D = np.reshape(trainX, (trainX.shape[0], timestep,1,1, dim))
# testX_input5D = np.reshape(testX, (testX.shape[0],timestep,1,1, dim))
print('构造得到模型的输入数据(训练数据已有标签trainY): ', trainX_input5D.shape, testX_input5D.shape)

# create and fit the convlstm network
if __name__ == '__main__':
    model = Sequential()
    model.add(ConvLSTM2D(filters=12, kernel_size=(2, 2), activation='relu',
                         input_shape=(2, 1, timestep // 2, dim),
                         #                         input_shape=(timestep,1,1, dim),
                         padding='same', return_sequences=True))
    model.add(ConvLSTM2D(filters=12, kernel_size=(2, 2), activation='relu',
                         padding='same', return_sequences=True))

    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX_input5D, trainY, epochs=1000)

    # 打印模型
    model.summary()

    # 开始预测
    trainPredict = model.predict(trainX_input5D)
    testPredict = model.predict(testX_input5D)

    # 逆缩放预测值
    dataframe = read_csv('../../XI-2/data/try2ed/PD-TVFEMD-BDX.csv', engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    Y = dataset[:, 0]
    trainPre = trainPredict * ((np.max(Y) - np.min(Y))) + np.min(Y)
    trainPre = trainPre.reshape(-1)
    trainPre = pd.DataFrame(trainPre)
    trainPre.to_csv('../../XI-2/data/pre_result/convlstmtrain.csv', header=False)

    testPre = testPredict * ((np.max(Y) - np.min(Y))) + np.min(Y)
    testPre = testPre.reshape(-1)
    dataframe = read_csv('../../XI-2/data/try2ed/QSX.csv', usecols=[0], engine='python')
    QSX = dataframe.values
    QSX = QSX.astype('float32')
    QSX = QSX[train_size + timestep:len(dataset)]
    QSX = QSX.reshape(-1)
    dataframe = read_csv('../../XI-2/data/try2ed/PD.csv', engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    Y = dataset[:, 0]
    Y = Y[train_size + timestep:len(dataset)]
    # testY_ori = testY*((np.max(dataset)-np.min(dataset)))+np.min(dataset)
    # testY_ori = testY_ori.reshape((-1,1))
    testY_ori = Y.reshape((-1, 1))
    for i in range(len(testY_ori)):
        testPredict[i] = QSX[i] + testPre[i]
    testPredict = testPredict.reshape(-1)

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
        error.append(testY_ori[i] - testPredict[i])
        error1.append(abs((testY_ori[i] - testPredict[i]) / testY_ori[i]))
        error2.append(testY_ori[i] * testPredict[i])
        squared1.append(testY_ori[i] * testY_ori[i])
        squared2.append(testPredict[i] * testPredict[i])

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
        IAerror.append((abs(testY_ori[i] - meannn) + abs(testPredict[i] - meannn)) * (
                    abs(testY_ori[i] - meannn) + abs(testPredict[i] - meannn)))

    U2error1 = []
    U2error2 = []
    for i in range(len(testY_ori) - 1):
        U2error1.append(((testY_ori[i + 1] - testPredict[i + 1]) / testY_ori[i]) * (
                    (testY_ori[i + 1] - testPredict[i + 1]) / testY_ori[i]))
        U2error2.append(
            ((testY_ori[i + 1] - testPredict[i]) / testY_ori[i]) * ((testY_ori[i + 1] - testPredict[i]) / testY_ori[i]))
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
    testPredict = testPredict.reshape(-1)
    testPredict = pd.DataFrame(testPredict)
    testPredict.to_csv('../../XI-2/data/pre_result/convlstmoutput.csv', header=False)
    import csv

    Evaluation_index = [MAE, RMSE, NMSE, MAPE, IA, U1index, U2index]
    with open("../../XI-2/data/pre_result/convlstmEvaluation.csv", "a", newline='',
              encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(Evaluation_index)

    plt.figure(facecolor='white')
    plt.plot(testPredict, color='green', label='Predict')
    plt.plot(testY_ori, color='pink', label='Original')
    plt.legend(loc='best')
    plt.show()
    pres.append(testPredict)

end = time.time()
print('Running time: %d seconds' % (end - start))