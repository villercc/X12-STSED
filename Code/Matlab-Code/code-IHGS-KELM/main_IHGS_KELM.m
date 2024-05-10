%_________________________________________________________________________%
%���ģ�����IHGS�㷨�Ż���KELM�ع�����       %
%Դ��by Jack�� �� https://mianbaoduo.com/o/bread/mbd-YZaTlppv
%_________________________________________________________________________%
clear all 
%clc
%% ��������
% load data
% % �������ѵ���������Լ�
% k = randperm(size(input,1));
% % ѵ��������1900������
% P_train=input(k(1:1900),:)';
% T_train=output(k(1:1900));
% % ���Լ�����100������
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

% AA=xlsread('ѵ������.xls');
% BB=xlsread('ѵ�����.xls');
% CC=xlsread('��������.xls');
% DD=xlsread('�������.xls');
% 
% P_train = AA';
% T_train = BB';
% 
% P_test = CC';
% T_test = DD';

%% ��һ��
% input
[Pn_train,inputps] = mapminmax(P_train,0,1);
Pn_test = mapminmax('apply',P_test,inputps);
% ouput
[Tn_train,outputps] = mapminmax(T_train,0,1);
Tn_test = mapminmax('apply',T_test,outputps);
%% ��������
N=30; % Number of search agents
FEs=10; % Maximum number of evaluation times
dim = 2;   %dimension size% ά��Ϊ2�����Ż���������������ϵ�� C �ͺ˺������� S
lb = [0.1,0.1];%�±߽�
ub = [80,10];%�ϱ߽�
fobj = @(x) fun(x,Pn_train,Tn_train);
[Destination_fitness,bestPositions,Convergence_curve]=IHGS(N,FEs,lb,ub,dim,fobj);%��ʼ�Ż�
figure
plot(Convergence_curve,'linewidth',1.5);
grid on;
xlabel('��������')
ylabel('��Ӧ��ֵ')
title('IHGS��������')
%% ��ȡ��������ϵ�� C �ͺ˺������� S
Regularization_coefficient = bestPositions(1);
Kernel_para = bestPositions(2);
Kernel_type = 'rbf';
%% ѵ��
[TrainOutT,OutputWeight] = kelmTrain(Pn_train,Tn_train,Regularization_coefficient,Kernel_type,Kernel_para);
%% ѵ����Ԥ��
InputWeight = OutputWeight;
[TestOutT] = kelmPredict(Pn_train,InputWeight,Kernel_type,Kernel_para,Pn_test);
%% ѵ������ȷ��
TrainOutT = mapminmax('reverse',TrainOutT,outputps);
errorTrain = TrainOutT - T_train;
MSEErrorTrain = mse(errorTrain);

%% ���Լ���ȷ��
TestOutT = mapminmax('reverse',TestOutT,outputps);
errorTest = TestOutT - T_test;
MSEErrorTest = mse(errorTest);


%% ���û���KELM����Ԥ��
Regularization_coefficient1 = 4;
Kernel_para1 = [2];                   %�˺�����������
Kernel_type = 'rbf';
%% ѵ��
[TrainOutT1,OutputWeight1] = kelmTrain(Pn_train,Tn_train,Regularization_coefficient1,Kernel_type,Kernel_para1);

%% Ԥ��
InputWeight1 = OutputWeight1;
[TestOutT1] = kelmPredict(Pn_train,InputWeight1,Kernel_type,Kernel_para1,Pn_test);

%% ѵ������ȷ��
TrainOutT1 = mapminmax('reverse',TrainOutT1,outputps);%����һ��
errorTrain1 = TrainOutT1 - T_train;
MSEErrorTrain1 = mse(errorTrain1);
%% ���Լ���ȷ��
TestOutT1 = mapminmax('reverse',TestOutT1,outputps);
errorTest1 = TestOutT1 - T_test;
MSEErrorTest1 = mse(errorTest1);

%����Ԥ��ͼ
figure
plot(T_train,'o-');
hold on
plot(TrainOutT,'*-');
plot(TrainOutT1,'.-')
title('ѵ�������')
legend('��ʵ���','IHGS-KELMԤ����','KELMԤ����');
grid on;
%�������ͼ
figure
plot(errorTrain,'r-','linewidth',1.5);
hold on;
plot(errorTrain1,'b-','linewidth',1.5)
legend('IHGS-KELM','KELM');
title('ѵ�������')
grid on


%% ���Լ���ȷ��
figure
plot(T_test,'o-');
hold on
plot(TestOutT,'*-');
plot(TestOutT1,'.-');
title('���Լ����')
legend('��ʵ���','IHGS-EKLMԤ����','EKLMԤ����');
grid on;


figure
plot(errorTest,'r-','linewidth',1.5);
hold on
plot(errorTest1,'b-','linewidth',1.5);
title('���Լ����');
legend('IHGS-EKLM','EKLM')
grid on

disp(['ѵ����IHGS-KELM ��MSE:',num2str(MSEErrorTrain)])
disp(['ѵ����KELM ��MSE:',num2str(MSEErrorTrain1)])
disp(['���Լ�IHGS-KELM ��MSE:',num2str(MSEErrorTest)])
disp(['���Լ�KELM ��MSE:',num2str(MSEErrorTest1)])











