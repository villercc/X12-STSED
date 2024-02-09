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
Max_iter=100; % Maximum number of evaluation times
dimSize = 5;   %dimension size


% Load details of the selected benchmark function
[lb,ub,dim,fobj]=Reconstruct_fun_IHGS(Function_name,dimSize,X_minmax_data,Y_minmax_data); 

%% IHGS
disp('IHGS is now tackling your problem')

tic
% initialize position
bestPositions=zeros(1,dim);
tempPosition=zeros(N,dim);

Destination_fitness=inf;%change this to -inf for maximization problems
Worstest_fitness=-inf;
AllFitness = inf*ones(N,1);%record the fitness of all positions
VC1 = ones(N,1);%record the variation control of all positions  ==E

weight3 = ones(N,dim);%hungry weight of each position
weight4 = ones(N,dim);%hungry weight of each position

%Initialize the set of random solutions
X=Circle_initialization(N,dim,ub,lb);
Convergence_curve=zeros(1,Max_iter);
it=1; %Number of iterations

hungry = zeros(1,size(X,1));%record the hungry of all positions
count=0;

% Main loop
while  it <= Max_iter
    VC2 = 0.03; %The variable of variation control ==Lowercase"L"

    sumHungry = 0;%record the sum of each hungry 
    
    %sort the fitness
    for i=1:size(X,1)
        % Check if solutions go outside the search space and bring them back
        Flag4ub=X(i,:)>ub;
        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        AllFitness(i) = fobj(X(i,:));
    end
    [AllFitnessSorted,IndexSorted] = sort(AllFitness);
    bestFitness = AllFitnessSorted(1);
    worstFitness = AllFitnessSorted(size(X,1));
    
    %update the best fitness value and best position
    if bestFitness < Destination_fitness
        bestPositions=X(IndexSorted(1),:);
        Destination_fitness = bestFitness;
        count=0;
    end
    
    if worstFitness > Worstest_fitness
        Worstest_fitness = worstFitness;
    end
    
    for i = 1:size(X,1)
         %calculate the variation control of all positions
         VC1(i) = sech(abs(AllFitness(i)-Destination_fitness));    
         %calculate the hungry of each position
        if Destination_fitness == AllFitness(i)
            hungry(1,i) = 0;
            count = count+1;
            tempPosition(count,:)=X(i,:);
        else
            temprand = rand();
            c = (AllFitness(i)-Destination_fitness)/(Worstest_fitness-Destination_fitness)*temprand*2*(ub-lb);
            if c<100   %LH=100
                b=100*(1+temprand);  
            else
                b=c;
            end   
            hungry(1,i) = hungry(1,i)+ max(b); 
            sumHungry = sumHungry + hungry(1,i);
        end
    end 
    
    %calculate the hungry weight of each position
    for i=1:size(X,1)
        for j=2:size(X,2)
                weight3(i,j) = (1-exp(-abs(hungry(1,i)-sumHungry)))*rand()*2;
                if rand()<VC2
                    weight4(i,j) = hungry(1,i)*size(X,1)/sumHungry*rand();
                else
                    weight4(i,j) = 1;
                end
        end
        
    end
    
    
    % Update the Position of search agents
    shrink=0.005+(2-0.005)*(2^(-10*((it/(Max_iter))^4))); % a decreases nonlinearly fron 2 to 0
    %shrink=2*(1-it/Max_iter); % a decreases linearly fron 2 to 0
    for i=1:size(X,1)
        if rand<VC2
            X(i,:) = X(i,j)*(1+randn(1)); %Game1
        else
            A = randi([1,count]);
            for j=1:size(X,2)
                r = rand();
                vb = 2*shrink*r-shrink;%[-a,a]
                % Moving based on the bestPosition
                % The transformation range is controlled by weight3,bestPositions and X
                if r>VC1(i)
                    X(i,j) = weight4(i,j)*tempPosition(A,j)+vb*weight3(i,j)*abs(tempPosition(A,j)-X(i,j));
                else
                    X(i,j) = weight4(i,j)*tempPosition(A,j)-vb*weight3(i,j)*abs(tempPosition(A,j)-X(i,j));
                end
            end
        end
    end
    
    Convergence_curve(it)=Destination_fitness;
    it=it+1;
end
toc