%%适应度函数
function  fitness = fun(x,Pn_train,Tn_train)
Regularization_coefficient = x(1);
Kernel_para = x(2);
Kernel_type = 'rbf';
%% 训练
[TrainOutT,OutputWeight] = kelmTrain(Pn_train,Tn_train,Regularization_coefficient,Kernel_type,Kernel_para);
%% 训练集正确率
error = TrainOutT - Tn_train;
fitness = mse(error);

end