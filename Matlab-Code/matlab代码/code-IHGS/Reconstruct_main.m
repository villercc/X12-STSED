%%
clear all 
clc

%% 导入数据
dataset = xlsread('C:\Users\Administrator\Desktop\XI-2\data\try2ed\PD-TVFEMD-BDX.csv');
%清除临时变量
clearvars raw;
%训练集（252）：测试集（72）=7：2
%train = dataset((1:252),:);
%test = dataset((253:324),:);
%训练集输入输出
%X_train = train(:,(1:5))';
%Y_train = train(:,6)';
%测试集输入输出
%X_test = test(:,(1:5))';
%Y_test = test(:,6)';
X_data = dataset(:,2:6)';
Y_data = dataset(:,1)';

%% 进行线性函数归一化
%按照每行来归一化
[X_minmax_data,s1] = mapminmax(X_data,0,1);
%disp('mapminmax归一化');
%disp(X_minmax_data);
%disp('mapminmax归一化索引');
%disp(s1);

Y_data_l=size(Y_data,2);
Y_minmax_data = zeros(1,Y_data_l);
for n=1:Y_data_l
Y_minmax_data(:,n)=(Y_data(:,n)-min(Y_data))/(max(Y_data)-min(Y_data));
end
%disp('线性归一化');
%disp(Y_minmax_data);

%% 算法
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