clear;
N=20; % Number of search agents
FEs=20; % Maximum number of evaluation times
dim = 2;   %dimension size% ά��Ϊ2�����Ż���������������ϵ�� C �ͺ˺������� S
lb = [0.01,0.01];%�±߽�
ub = [50,10];%�ϱ߽�
X=Circle_initialization(N,dim,ub,lb);