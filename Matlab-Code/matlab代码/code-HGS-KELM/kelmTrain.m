%KELM （分类）训练函数
%输入：P_train训练输入数据
%      T_train训练数据对应的类别标签
%      Regularization_coefficient：正则化系数C
%      Kernel_type 核类型                - Type of Kernels:
%                                   'RBF_kernel' for RBF Kernel
%                                   'lin_kernel' for Linear Kernel
%                                   'poly_kernel' for Polynomial Kernel
%                                   'wav_kernel' for Wavelet Kernel
%
%       Kernel_para 核参数[0.1,10]
%输出： TrainOutT，训练以后，得到得预测标签值
%       OutputWeight，训练得到得权重
function [TrainOutT,OutputWeight] = kelmTrain(P_train,T_train,Regularization_coefficient,Kernel_type,Kernel_para)
%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = P_train;
n = length(T_train);
C = Regularization_coefficient;
[maxClass] = max(T_train);
T = T_train;
Omega_train = kernel_matrix(P',Kernel_type, Kernel_para);
OutputWeight=((Omega_train+speye(n)/C)\(T'));
Y=(Omega_train * OutputWeight)';
TrainOutT = Y;
end