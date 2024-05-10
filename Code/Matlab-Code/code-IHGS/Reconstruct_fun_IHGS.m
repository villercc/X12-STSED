%% 适应度函数
function [lb,ub,dim,fobj] = Reconstruct_fun_IHGS(F,dimSize,X_minmax_data,Y_minmax_data)

switch F
    case 'F1'
        fobj = @(x) F1(x,X_minmax_data,Y_minmax_data);
        lb = 0;%下边界
        ub = 1;%上边界
        dim = 5;
end

end

% F1

function o = F1(x,X_minmax_data,Y_minmax_data)
error = zeros(1,size(Y_minmax_data,2));
for i=1:size(Y_minmax_data,2)
    error(1:i) = Y_minmax_data(1,i)-(x(1)*X_minmax_data(1,i)+x(2)*X_minmax_data(2,i)+x(3)*X_minmax_data(3,i)+x(4)*X_minmax_data(4,i)+x(5)*X_minmax_data(5,i));
end
o = mse(error)^0.5;
end