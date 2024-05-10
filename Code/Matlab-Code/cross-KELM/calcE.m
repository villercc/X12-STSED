function Re = calcE(dc,dr)
% 作者：艾木樨
% 邮箱：moveon5@163.com
% 欢迎关注matlab爱好者微信公众号，请多多分享公众号文章，谢谢支持！
%——————————————————————————————————————%
%                              程序简介                                  %
% 本程序用于计算误差分析中相关的参数指标，计算结果采用结构体形式进行返回；
% 程序参数说明
% dc为计算值，dr为实际值
% Re.SSE —— 残差平方和
% Re.MSE —— 均方差
% Re.RMSE —— 均方根差
% Re.MAE —— 平均绝对误差
% Re.R2 —— 决定系数R方
% Re.COR —— 相关系数
% Re.MAPE —— 平均绝对百分误差
% Re.Theil —— 希尔不等系数
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
    % 计算R-squared
    rr = dc-dr;
    normr = norm(rr);
    SSE = normr.^2;
    SST = norm(dc-mean(dc))^2;
    Re.R2 = 1 - SSE/SST;
    % Re.COR —— 相关系数
    xm = mean(dc);
    tm = mean(dr);
    fz = sum((dc-xm).*(dr-tm));
    fm = sqrt(sum((dc-xm).^2)*sum((dr-tm).^2));
    Re.COR = fz/fm;
    Re.Theil = sqrt(sum((dc-dr).^2)/len)/(sqrt(sum(dr.^2)) + sqrt(sum(dc.^2)));
    fprintf(['残差平方和(SSE):',strcat('\t',num2str(Re.SSE)),'\r\n']);
    fprintf(['均方差(MSE):',strcat('\t',num2str(Re.MSE)),'\r\n']);
    fprintf(['均方根差(RMSE):',strcat('\t',num2str(Re.RMSE)),'\r\n']);
    fprintf(['平均绝对误差(MAE):',strcat('\t',num2str(Re.MAE)),'\r\n']);
    fprintf(['平均绝对百分误差(MAPE):',strcat('\t',num2str(Re.MAPE)),'\r\n']);
    fprintf(['决定系数R方(R2):',strcat('\t',num2str(Re.R2)),'\r\n']);
    fprintf(['相关系数(COR):',strcat('\t',num2str(Re.COR)),'\r\n']);
    fprintf(['希尔不等系数(Theil):',strcat('\t',num2str(Re.Theil)),'\r\n']);
else
    disp('数据格式必须为行或列数组！！！');
    Re = [];
end


