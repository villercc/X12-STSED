clc;
clear;
%% ----decomposition
%x=0:323
% for i=1:90
%     imf5erwu=tvf_emd(need(:,i))%第i列 会变成5行
%     QuzaoVar5erwu(i,:)=sum(imf5erwu(2:5,:),1);%行操作
% end
% QuzaoVar5erwu=QuzaoVar5erwu'%行转置为列
%% 导入数据
datatodecompose = xlsread('C:\Users\Administrator\Desktop\XI-2\data\try2ed\xiangguanxinghao5.xlsx','Sheet1');

%% 清除临时变量
clearvars raw;
%% 分解
konghang = zeros(1,size(datatodecompose,1));
ALLimf = zeros(1,size(datatodecompose,1));
ALLquzao = zeros(1,size(datatodecompose,1));
selected_imf = zeros(1,size(datatodecompose,1));%行向量(1,324)
for k=1:5
    %uname = ['DANYI_',num2str(k),'_imf'];
    DANYI_imf = tvf_emd(datatodecompose(:,k));%导入数据为几组列向量，分解后数据为n组行向量
    DANYI_quzao = zeros(1,size(datatodecompose,1));
    for i=2:size(DANYI_imf,1)
        DANYI_quzao = DANYI_quzao+DANYI_imf(i,:);%第2行加到最后一行
    end
    ALLquzao = [ALLquzao;DANYI_quzao];
    ALLimf = [ALLimf;DANYI_imf;konghang];
end
ALLimf = ALLimf';
ALLquzao = ALLquzao';
