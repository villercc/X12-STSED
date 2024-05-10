%% 导入电子表格中的数据
% 用于从以下电子表格导入数据的脚本:
%
%    工作簿: C:\Users\Administrator\Desktop\need分解数据.xlsx
%    工作表: Sheet1
%
% 要扩展代码以供其他选定数据或其他电子表格使用，请生成函数来代替脚本。

% 由 MATLAB 自动生成于 2021/12/28 21:34:43

%% 导入数据
data = xlsread('C:\Users\Administrator\Desktop\need分解数据.xlsx','Sheet1');

%% 将导入的数组分配给列变量名称
VarName1 = data(:,1);
VarName2 = data(:,2);
VarName3 = data(:,3);
VarName4 = data(:,4);
VarName5 = data(:,5);
VarName6 = data(:,6);
VarName7 = data(:,7);
VarName8 = data(:,8);
VarName9 = data(:,9);
VarName10 = data(:,10);
VarName11 = data(:,11);
VarName12 = data(:,12);
VarName13 = data(:,13);
VarName14 = data(:,14);
VarName15 = data(:,15);
VarName16 = data(:,16);
VarName17 = data(:,17);
VarName18 = data(:,18);
VarName19 = data(:,19);
VarName20 = data(:,20);
VarName21 = data(:,21);
VarName22 = data(:,22);
VarName23 = data(:,23);
VarName24 = data(:,24);
VarName25 = data(:,25);
VarName26 = data(:,26);
VarName27 = data(:,27);
VarName28 = data(:,28);
VarName29 = data(:,29);
VarName30 = data(:,30);
VarName31 = data(:,31);
VarName32 = data(:,32);
VarName33 = data(:,33);
VarName34 = data(:,34);
VarName35 = data(:,35);
VarName36 = data(:,36);
VarName37 = data(:,37);
VarName38 = data(:,38);
VarName39 = data(:,39);
VarName40 = data(:,40);
VarName41 = data(:,41);
VarName42 = data(:,42);
VarName43 = data(:,43);
VarName44 = data(:,44);
VarName45 = data(:,45);
VarName46 = data(:,46);
VarName47 = data(:,47);
VarName48 = data(:,48);
VarName49 = data(:,49);
VarName50 = data(:,50);
VarName51 = data(:,51);
VarName52 = data(:,52);
VarName53 = data(:,53);
VarName54 = data(:,54);
VarName55 = data(:,55);
VarName56 = data(:,56);
VarName57 = data(:,57);
VarName58 = data(:,58);
VarName59 = data(:,59);
VarName60 = data(:,60);
VarName61 = data(:,61);
VarName62 = data(:,62);
VarName63 = data(:,63);
VarName64 = data(:,64);
VarName65 = data(:,65);
VarName66 = data(:,66);
VarName67 = data(:,67);
VarName68 = data(:,68);
VarName69 = data(:,69);
VarName70 = data(:,70);
VarName71 = data(:,71);
VarName72 = data(:,72);
VarName73 = data(:,73);
VarName74 = data(:,74);
VarName75 = data(:,75);
VarName76 = data(:,76);
VarName77 = data(:,77);
VarName78 = data(:,78);
VarName79 = data(:,79);
VarName80 = data(:,80);
VarName81 = data(:,81);
VarName82 = data(:,82);
VarName83 = data(:,83);
VarName84 = data(:,84);
VarName85 = data(:,85);
VarName86 = data(:,86);
VarName87 = data(:,87);
VarName88 = data(:,88);
VarName89 = data(:,89);
VarName90 = data(:,90);

%% 清除临时变量
clearvars data raw;