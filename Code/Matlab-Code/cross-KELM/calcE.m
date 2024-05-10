function Re = calcE(dc,dr)
% ���ߣ���ľ��
% ���䣺moveon5@163.com
% ��ӭ��עmatlab������΢�Ź��ںţ���������ں����£�лл֧�֣�
%����������������������������������������������������������������������������%
%                              ������                                  %
% ���������ڼ�������������صĲ���ָ�꣬���������ýṹ����ʽ���з��أ�
% �������˵��
% dcΪ����ֵ��drΪʵ��ֵ
% Re.SSE ���� �в�ƽ����
% Re.MSE ���� ������
% Re.RMSE ���� ��������
% Re.MAE ���� ƽ���������
% Re.R2 ���� ����ϵ��R��
% Re.COR ���� ���ϵ��
% Re.MAPE ���� ƽ�����԰ٷ����
% Re.Theil ���� ϣ������ϵ��
[mc,nc] = size(dc);
[mr,nr] = size(dr);
maxc = max([mc,nc]);
minc = min([mc,nc]);
maxr = max([mr,nr]);
minr = min([mr,nr]);
if (maxc==maxr && minc==minr) && minc == 1
    if mc ~= mr
        dr = dr';
    end
    len = length(dr);
    Re.SSE = sum((dc-dr).^2);
    Re.MSE = sum((dc-dr).^2)/len;
    Re.RMSE = sqrt(sum((dc-dr).^2)/len);
    Re.MAE = sum(abs(dc-dr))/len;
    Re.MAPE = 100*sum(abs(dc-dr)./dr)/len;
    % ����R-squared
    rr = dc-dr;
    normr = norm(rr);
    SSE = normr.^2;
    SST = norm(dc-mean(dc))^2;
    Re.R2 = 1 - SSE/SST;
    % Re.COR ���� ���ϵ��
    xm = mean(dc);
    tm = mean(dr);
    fz = sum((dc-xm).*(dr-tm));
    fm = sqrt(sum((dc-xm).^2)*sum((dr-tm).^2));
    Re.COR = fz/fm;
    Re.Theil = sqrt(sum((dc-dr).^2)/len)/(sqrt(sum(dr.^2)) + sqrt(sum(dc.^2)));
    fprintf(['�в�ƽ����(SSE):',strcat('\t',num2str(Re.SSE)),'\r\n']);
    fprintf(['������(MSE):',strcat('\t',num2str(Re.MSE)),'\r\n']);
    fprintf(['��������(RMSE):',strcat('\t',num2str(Re.RMSE)),'\r\n']);
    fprintf(['ƽ���������(MAE):',strcat('\t',num2str(Re.MAE)),'\r\n']);
    fprintf(['ƽ�����԰ٷ����(MAPE):',strcat('\t',num2str(Re.MAPE)),'\r\n']);
    fprintf(['����ϵ��R��(R2):',strcat('\t',num2str(Re.R2)),'\r\n']);
    fprintf(['���ϵ��(COR):',strcat('\t',num2str(Re.COR)),'\r\n']);
    fprintf(['ϣ������ϵ��(Theil):',strcat('\t',num2str(Re.Theil)),'\r\n']);
else
    disp('���ݸ�ʽ����Ϊ�л������飡����');
    Re = [];
end


