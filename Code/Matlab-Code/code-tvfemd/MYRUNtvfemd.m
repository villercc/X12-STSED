clc;
clear;
%% ----decomposition
%x=0:323
% for i=1:90
%     imf5erwu=tvf_emd(need(:,i))%��i�� ����5��
%     QuzaoVar5erwu(i,:)=sum(imf5erwu(2:5,:),1);%�в���
% end
% QuzaoVar5erwu=QuzaoVar5erwu'%��ת��Ϊ��
%% ��������
datatodecompose = xlsread('C:\Users\Administrator\Desktop\XI-2\data\try2ed\xiangguanxinghao5.xlsx','Sheet1');

%% �����ʱ����
clearvars raw;
%% �ֽ�
konghang = zeros(1,size(datatodecompose,1));
ALLimf = zeros(1,size(datatodecompose,1));
ALLquzao = zeros(1,size(datatodecompose,1));
selected_imf = zeros(1,size(datatodecompose,1));%������(1,324)
for k=1:5
    %uname = ['DANYI_',num2str(k),'_imf'];
    DANYI_imf = tvf_emd(datatodecompose(:,k));%��������Ϊ�������������ֽ������Ϊn��������
    DANYI_quzao = zeros(1,size(datatodecompose,1));
    for i=2:size(DANYI_imf,1)
        DANYI_quzao = DANYI_quzao+DANYI_imf(i,:);%��2�мӵ����һ��
    end
    ALLquzao = [ALLquzao;DANYI_quzao];
    ALLimf = [ALLimf;DANYI_imf;konghang];
end
ALLimf = ALLimf';
ALLquzao = ALLquzao';
