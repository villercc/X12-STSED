%%
clear all 
clc

%% ��������
dataset = xlsread('C:\Users\Administrator\Desktop\XI-2\data\try2ed\PD-TVFEMD-BDX.csv');
%�����ʱ����
clearvars raw;
%ѵ������252�������Լ���72��=7��2
%train = dataset((1:252),:);
%test = dataset((253:324),:);
%ѵ�����������
%X_train = train(:,(1:5))';
%Y_train = train(:,6)';
%���Լ��������
%X_test = test(:,(1:5))';
%Y_test = test(:,6)';
X_data = dataset(:,2:6)';
Y_data = dataset(:,1)';

%% �������Ժ�����һ��
%����ÿ������һ��
[X_minmax_data,s1] = mapminmax(X_data,0,1);
%disp('mapminmax��һ��');
%disp(X_minmax_data);
%disp('mapminmax��һ������');
%disp(s1);

Y_data_l=size(Y_data,2);
Y_minmax_data = zeros(1,Y_data_l);
for n=1:Y_data_l
Y_minmax_data(:,n)=(Y_data(:,n)-min(Y_data))/(max(Y_data)-min(Y_data));
end
%disp('���Թ�һ��');
%disp(Y_minmax_data);

%% �㷨
N=30; % Number of search agents
Function_name='F1'; 
FEs=100; % Maximum number of evaluation times
dimSize = 5;   %dimension size


% Load details of the selected benchmark function
[lb,ub,dim,fobj]=Reconstruct_fun_IHGS(Function_name,dimSize,X_minmax_data,Y_minmax_data); 

[Destination_fitness,bestPositions,Convergence_curve]=Reconstruct_IHGS(N,FEs,lb,ub,dim,fobj);


%Draw objective space
figure,
hold on
semilogy(Convergence_curve,'Color','b','LineWidth',4);
title('Convergence curve')
xlabel('Iteration');
ylabel('Best fitness obtained so far');
axis tight
grid off
box on
legend('IHGS')


display(['The best location of IHGS is: ', num2str(bestPositions)]);
display(['The best fitness of IHGS is: ', num2str(Destination_fitness)]);