%KELM （分类）预测函数
%输入：P_train训练输入数据
%      InputWeight:训练以后得到的权重
%      Kernel_type 核类型                - Type of Kernels:
%                                   'RBF_kernel' for RBF Kernel
%                                   'lin_kernel' for Linear Kernel
%                                   'poly_kernel' for Polynomial Kernel
%                                   'wav_kernel' for Wavelet Kernel
%
%       Kernel_para 核参数[0.1,10]
%       测试集输入
%输出： TestOutT，预测得到的结果
function [TestOutT] = kelmPredict(P_train,InputWeight,Kernel_type,Kernel_para,P_test)

P = P_train;
PT = P_test;
Omega_test = kernel_matrix(P',Kernel_type, Kernel_para,PT');
TY=(Omega_test' * InputWeight)';                            %   TY: the actual output of the testing data
TestOutT = TY;
end