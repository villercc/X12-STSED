%_________________________________________________________________________%
%基于遗传算法优化的KELM回归问题       %
%by Jack旭 ： https://mianbaoduo.com/o/bread/mbd-YZaTlppv
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
% 训练集
[Pn_train,inputps] = mapminmax(P_train,-1,1);
Pn_test = mapminmax('apply',P_test,inputps);
% 测试集
[Tn_train,outputps] = mapminmax(T_train,-1,1);
Tn_test = mapminmax('apply',T_test,outputps);
%% 遗传参数设置
pop=30; %种群数量
Max_iteration=10; %  设定最大迭代次数
dim = 2;% 维度为2，即优化两个参数，正则化系数 C 和核函数参数 S
lb = [0.1;0.1];%下边界
ub = [50;10];%上边界
fobj = @(x) fun(x,Pn_train,Tn_train);
[Best_score,Best_pos,GA_curve]=GA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化
figure
plot(GA_curve,'linewidth',1.5);
grid on;
xlabel('迭代次数')
ylabel('适应度值')
title('遗传收敛曲线')
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
legend('真实类别','GA-KELM预测结果','KELM预测结果');
grid on;
%绘制误差图
figure
plot(errorTrain,'r-','linewidth',1.5);
hold on;
plot(errorTrain1,'b-','linewidth',1.5)
legend('GA - KELM','KELM');
title('训练集误差')
grid on


%% 测试集正确率
figure
plot(T_test,'o-');
hold on
plot(TestOutT,'*-');
plot(TestOutT1,'.-');
title('测试集结果')
legend('真实类别','GA-EKLM预测结果','EKLM预测结果');
grid on;


figure
plot(errorTest,'r-','linewidth',1.5);
hold on
plot(errorTest1,'b-','linewidth',1.5);
title('测试集误差');
legend('GA-EKLM','EKLM')
grid on

disp(['训练集GA-KELM 的MSE:',num2str(MSEErrorTrain)])
disp(['训练集KELM 的MSE:',num2str(MSEErrorTrain1)])
disp(['测试集GA-KELM 的MSE:',num2str(MSEErrorTest)])
disp(['测试集KELM 的MSE:',num2str(MSEErrorTest1)])











