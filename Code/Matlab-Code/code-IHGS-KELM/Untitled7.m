clear;
N=20; % Number of search agents
FEs=20; % Maximum number of evaluation times
dim = 2;   %dimension size% 维度为2，即优化两个参数，正则化系数 C 和核函数参数 S
lb = [0.01,0.01];%下边界
ub = [50,10];%上边界
X=Circle_initialization(N,dim,ub,lb);