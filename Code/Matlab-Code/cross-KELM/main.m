close all;clear all;
clc
% AA=xlsread('/Users/phil/Desktop/����ѵ������.xls');
% BB=xlsread('/Users/phil/Desktop/����ѵ�����.xls');
% CC=xlsread('/Users/phil/Desktop/�����������.xls');
% DD=xlsread('/Users/phil/Desktop/����������.xls');

AA=xlsread('ѵ������.xls');
BB=xlsread('ѵ�����.xls');
CC=xlsread('��������.xls');
DD=xlsread('�������.xls');

P_train = AA';
T_train = BB';

P_test = CC';
T_test = DD';
%% ���ݹ�һ��

% ѵ����
[Pn_train,inputps] = mapminmax(P_train,0,1);
Pn_test = mapminmax('apply',P_test,inputps);
% ���Լ�
[Tn_train,outputps] = mapminmax(T_train,0,1);
Tn_test = mapminmax('apply',T_test,outputps);

% ����/ѵ��kelm 

[OutputWeight,Omega_test,Y] = kelmtrain(Pn_train, Tn_train, Pn_test, 0, 1000, 'RBF_kernel',10);
A=Y;
%B = wucha(Pn_train,Tn_train, 0, 1000, 'RBF_kernel',10)

P_sim=mapminmax('reverse',Y,outputps);
tn_sim = kelmpredict(OutputWeight,Omega_test);
T_sim = mapminmax('reverse',tn_sim,outputps)

%% ����Ա�
result = [T_test' T_sim'];
% �������
E = mse(T_sim - T_test);
% ����ϵ��
N = length(T_test);
R2=(N*sum(T_sim.*T_test)-sum(T_sim)*sum(T_test))^2/((N*sum((T_sim).^2)-(sum(T_sim))^2)*(N*sum((T_test).^2)-(sum(T_test))^2)); 

%% ��ͼ
figure(1)
plot(1:N,T_test,'r-*',1:N,T_sim,'b:o')
grid on
legend('��ʵֵ','Ԥ��ֵ')
xlabel('�������')
ylabel('�ۿ�������')
string = {'Ԥ�����Ա�(KELM)';['(mse = ' num2str(E) ' R^2 = ' num2str(R2) ')']};
title(string)

Re=calcE(T_sim,T_test);
