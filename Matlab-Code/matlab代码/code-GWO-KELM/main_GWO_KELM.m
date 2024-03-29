%_________________________________________________________________________%
%更改：基于GWO算法优化的KELM回归问题       %
%源码by Jack旭 ： https://mianbaoduo.com/o/bread/mbd-YZaTlppv
%_________________________________________________________________________%
clear all 
%clc
%% 导入数据
% load data
% % 随机生成训练集、测试集
% k = randperm(size(input,1));
% % 训练集——1900个样本
% P_train=input(k(1:1900),:)';
% T_train=output(k(1:1900));
% % 测试集——100个样本
% P_test=input(k(1901:2000),:)';
% T_test=output(k(1901:2000));

AA=xlsread('model_train_input.xlsx');
BB=xlsread('model_train_output.xlsx');
CC=xlsread('model_test_input.xlsx');
DD=xlsread('model_test_output.xlsx');

P_train = CC';
T_train = DD';

P_test = CC';
T_test = DD';

% AA=xlsread('训练输入.xls');
% BB=xlsread('训练输出.xls');
% CC=xlsread('测试输入.xls');
% DD=xlsread('测试输出.xls');
% 
% P_train = AA';
% T_train = BB';
% 
% P_test = CC';
% T_test = DD';

%% 归一化
% input
[Pn_train,inputps] = mapminmax(P_train,0,1);
Pn_test = mapminmax('apply',P_test,inputps);
% ouput
[Tn_train,outputps] = mapminmax(T_train,0,1);
Tn_test = mapminmax('apply',T_test,outputps);
%% 参数设置
SearchAgents_no=30; % Number of search agents
Max_iteration=20; % Maximum numbef of iterations
dim = 2;   %dimension size% 维度为2，即优化两个参数，正则化系数 C 和核函数参数 S
lb = [0.1,0.1];%下边界
ub = [50,10];%上边界
fobj = @(x) fun(x,Pn_train,Tn_train);
[Best_score,Best_pos,GWO_cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);%开始优化

figure
plot(GWO_cg_curve,'linewidth',1.5);
grid on;
xlabel('迭代次数')
ylabel('适应度值')
title('IHGS收敛曲线')
%% 获取最优正则化系数 C 和核函数参数 S
Regularization_coefficient = Best_pos(1);
Kernel_para = Best_pos(2);
Kernel_type = 'rbf';
%% 训练
[TrainOutT,OutputWeight] = kelmTrain(Pn_train,Tn_train,Regularization_coefficient,Kernel_type,Kernel_para);
%% 训练集预测
InputWeight = OutputWeight;
[TestOutT] = kelmPredict(Pn_train,InputWeight,Kernel_type,Kernel_para,Pn_test);
%% 训练集正确率
TrainOutT = mapminmax('reverse',TrainOutT,outputps);
errorTrain = TrainOutT - T_train;
MSEErrorTrain = mse(errorTrain);

%% 测试集正确率
TestOutT = mapminmax('reverse',TestOutT,outputps);
errorTest = TestOutT - T_test;
MSEErrorTest = mse(errorTest);


%% 利用基础KELM进行预测
Regularization_coefficient1 = 4;
Kernel_para1 = [2];                   %核函数参数矩阵
Kernel_type = 'rbf';
%% 训练
[TrainOutT1,OutputWeight1] = kelmTrain(Pn_train,Tn_train,Regularization_coefficient1,Kernel_type,Kernel_para1);

%% 预测
InputWeight1 = OutputWeight1;
[TestOutT1] = kelmPredict(Pn_train,InputWeight1,Kernel_type,Kernel_para1,Pn_test);

%% 训练集正确率
TrainOutT1 = mapminmax('reverse',TrainOutT1,outputps);%反归一化
errorTrain1 = TrainOutT1 - T_train;
MSEErrorTrain1 = mse(errorTrain1);
%% 测试集正确率
TestOutT1 = mapminmax('reverse',TestOutT1,outputps);
errorTest1 = TestOutT1 - T_test;
MSEErrorTest1 = mse(errorTest1);

%绘制预测图
figure
plot(T_train,'o-');
hold on
plot(TrainOutT,'*-');
plot(TrainOutT1,'.-')
title('训练集结果')
legend('真实类别','GWO-KELM预测结果','KELM预测结果');
grid on;
%绘制误差图
figure
plot(errorTrain,'r-','linewidth',1.5);
hold on;
plot(errorTrain1,'b-','linewidth',1.5)
legend('GWO-KELM','KELM');
title('训练集误差')
grid on


%% 测试集正确率
figure
plot(T_test,'o-');
hold on
plot(TestOutT,'*-');
plot(TestOutT1,'.-');
title('测试集结果')
legend('真实类别','GWO-EKLM预测结果','EKLM预测结果');
grid on;


figure
plot(errorTest,'r-','linewidth',1.5);
hold on
plot(errorTest1,'b-','linewidth',1.5);
title('测试集误差');
legend('GWO-EKLM','EKLM')
grid on

disp(['训练集GWO-KELM 的MSE:',num2str(MSEErrorTrain)])
disp(['训练集KELM 的MSE:',num2str(MSEErrorTrain1)])
disp(['测试集GWO-KELM 的MSE:',num2str(MSEErrorTest)])
disp(['测试集KELM 的MSE:',num2str(MSEErrorTest1)])











