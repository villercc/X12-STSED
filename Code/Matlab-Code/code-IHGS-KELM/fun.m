%%��Ӧ�Ⱥ���
function  fitness = fun(x,Pn_train,Tn_train)
Regularization_coefficient = x(1);
Kernel_para = x(2);
Kernel_type = 'rbf';
%% ѵ��
[TrainOutT,OutputWeight] = kelmTrain(Pn_train,Tn_train,Regularization_coefficient,Kernel_type,Kernel_para);
%% ѵ������ȷ��
error = TrainOutT - Tn_train;
fitness = mse(error);

end