function  [Px_train,Ty_train,Px_test,Ty_test] = createdata(timestep,Pn_train,Tn_train,Pn_test,Tn_test)
Px_train = Pn_train(:,1:timestep);
Ty_train = Tn_train(timestep+1);
for i = 2:(size(Pn_train,2)-timestep)
    a1 = Pn_train(:,i:(i+timestep-1));
    b1 = Tn_train(i+timestep);
    Px_train = cat(3,Px_train,a1);
    Ty_train = [Ty_train,b1];
end

Px_test = Pn_test(:,1:timestep);
Ty_test = Tn_test(timestep+1);
for i = 2:(size(Pn_test,2)-timestep)
    a2 = Pn_test(:,i:(i+timestep-1));
    b2 = Tn_test(i+timestep);
    Px_test = cat(3,Px_test,a2);
    Ty_test = [Ty_test,b2];
end
end