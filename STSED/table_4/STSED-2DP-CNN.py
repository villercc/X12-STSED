import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# matplotlib inline
import time

# fix random seed for reproducibility
np.random.seed(7)

start = time.time()

# load the dataset
dataframe = read_csv('../../XI-2/data/try2ed/PD-TVFEMD-BDX.csv', usecols=[0], engine='python')
# print(dataframe)
print("数据集的长度：", len(dataframe))
dataset = dataframe.values
# 将整型变为float
dataset = dataset.astype('float32')
# print(dataset)


# 数据缩放
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# print(X_scaler)
# print(Y_scaler)
# print(Y)


# 将数据拆分成训练和测试，7/9作为训练数据
train_size = int(len(dataset) * 0.65)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("原始训练集的长度：", train_size)
print("原始测试集的长度：", test_size)


# print(trainX)
# print(trainY)
# print(testX)
# print(testY)


# 重构数据集
##timestep为时间步长
def create_dataset(seq, timestep):
    dataX, dataY = [], []
    for i in range(len(seq) - timestep):
        a = seq[i:(i + timestep)]
        # X按照顺序取值 每次在后面增加一个数据
        dataX.append(a)
        # Y向后移动一位取值
        dataY.append(seq[i + timestep])
    return np.array(dataX), np.array(dataY)


# ----------forecasting-----------
timestep = 6
dim = 1
pres = []
trainX, trainY = create_dataset(train, timestep)
testX, testY = create_dataset(test, timestep)
print("转为监督学习，训练集数据长度：", len(trainX))
# print(trainX,trainY)
print("转为监督学习，测试集数据长度：", len(testX))
# print(testX, testY )


# 数据重构为4D [samples, subsequences, timesteps, features]
trainX_input4D = np.reshape(trainX, (trainX.shape[0], 1, timestep, dim))
testX_input4D = np.reshape(testX, (testX.shape[0], 1, timestep, dim))
print('构造得到模型的输入数据(训练数据已有标签trainY): ', trainX_input4D.shape, testX_input4D.shape)

# create and fit the convlstm network
if __name__ == '__main__':
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=12, kernel_size=3, activation='relu', input_shape=(None, 2, testX.shape[1]))))
    model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    model.add(TimeDistributed(Flatten()))

    # model.add(LSTM(4,activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX_input4D, trainY, epochs=1000)

    # 打印模型
    model.summary()

    # 开始预测
    trainPredict = model.predict(trainX_input4D)
    testPredict = model.predict(testX_input4D)

    # 逆缩放预测值
    dataframe = read_csv('../../XI-2/data/try2ed/PD-TVFEMD-BDX.csv', usecols=[0],
                         engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # trainPre = trainPredict*((np.max(dataset)-np.min(dataset)))+np.min(dataset)
    # trainPre = trainPre.reshape(-1)
    testPre = testPredict * ((np.max(dataset) - np.min(dataset))) + np.min(dataset)
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
    # trainScore = math.sqrt(mean_squared_error(trainY_re[:,0], trainPredict[:,0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(testY_re[:,0], testPredict[:,0]))
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
    testPredict.to_csv('../XI-2/data/pre_result/cnnoutput.csv', header=False)
    import csv

    Evaluation_index = [MAE, RMSE, NMSE, MAPE, IA, U1index, U2index]
    with open("../../XI-2/data/pre_result/cnnEvaluation.csv", "a", newline='',
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
