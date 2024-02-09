close all;clear all;
clc
% AA=xlsread('/Users/phil/Desktop/宝马训练输入.xls');
% BB=xlsread('/Users/phil/Desktop/宝马训练输出.xls');
% CC=xlsread('/Users/phil/Desktop/宝马测试输入.xls');
% DD=xlsread('/Users/phil/Desktop/宝马测试输出.xls');

AA=xlsread('训练输入.xls');
BB=xlsread('训练输出.xls');
CC=xlsread('测试输入.xls');
DD=xlsread('测试输出.xls');

P_train = AA';
T_train = BB';

P_test = CC';
T_test = DD';
%% 数据归一化

% 训练集
[Pn_train,inputps] = mapminmax(P_train,0,1);
Pn_test = mapminmax('apply',P_test,inputps);
% 测试集
[Tn_train,outputps] = mapminmax(T_train,0,1);
Tn_test = mapminmax('apply',T_test,outputps);

% 创建/训练kelm 

[OutputWeight,Omega_test,Y] = kelmtrain(Pn_train, Tn_train, Pn_test, 0, 1000, 'RBF_kernel',10);
A=Y;
%B = wucha(Pn_train,Tn_train, 0, 1000, 'RBF_kernel',10)

P_sim=mapminmax('reverse',Y,outputps);
tn_sim = kelmpredict(OutputWeight,Omega_test);
T_sim = mapminmax('reverse',tn_sim,outputps)

%% 结果对比
result = [T_test' T_sim'];
% 均方误差
E = mse(T_sim - T_test);
% 决定系数
N = length(T_test);
R2=(N*sum(T_sim.*T_test)-sum(T_sim)*sum(T_test))^2/((N*sum((T_sim).^2)-(sum(T_sim))^2)*(N*sum((T_test).^2)-(sum(T_test))^2)); 

%% 绘图
figure(1)
plot(1:N,T_test,'r-*',1:N,T_sim,'b:o')
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('港口吞吐量')
string = {'预测结果对比(KELM)';['(mse = ' num2str(E) ' R^2 = ' num2str(R2) ')']};
title(string)

Re=calcE(T_sim,T_test);
