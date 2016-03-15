function [ParamN_old] = ...
         gradDes(data_raw,data_GTT,label,a_u,...
		 ParamN_old,active_set_normal,mode)

%initial step size and tolerance level
step_size = 1e-3;
reltol = 1e-6;

%initialize inter-normal parameters
num_active_normal = sum(active_set_normal);
data_GTT_tempN = 0;
data_tempN = 0;
label_tempN = 0;
L2 = 0;
ParamN_new = ParamN_old;
if num_active_normal>=2
    tempN = find(active_set_normal==1);
    %first,filter out rare samples
    data_tempN = data_raw(label~=1,:);
    data_GTT_tempN = data_GTT(label~=1,:);
    label_tempN = label(label~=1);
end
Fs_multiclassN = 0;
if num_active_normal>=2
    Fs_multiclassN = ...
    getF_multiclass(data_tempN,data_GTT_tempN,label_tempN,...
		    ParamN_old,active_set_normal);
    for i=1:length(tempN)-1
        L2 = L2 + ...
	     norm([ParamN_old(tempN(i)).beta0 ParamN_old(tempN(i)).beta])^2;
    end
end

%calculate initial objective function value
Dold =  ObjF_meta(Fs_multiclassN,L2,...
		  data_GTT_tempN,label_tempN,...
		  label,active_set_normal,...
		  a_u,mode);
%disp(Dold);
Dnew = inf;
updated = true;
mu = step_size;
count_step = 0;
while abs(Dnew-Dold)>=reltol||Dnew>Dold||isinf(Dold) 

    count_step = count_step+1;
    fprintf('iteration %d; diff is %g\n',count_step,abs(Dnew-Dold));
    if updated==true
       temp_normal = 0;    
       if num_active_normal>=2
         Fs_multiclassN = ...
         getF_multiclass(data_tempN,data_GTT_tempN,label_tempN,...
         ParamN_old,active_set_normal);
       end
    
       if num_active_normal>=2
         %update parameters for each class           
         %initialize Dbeta0 and Dbeta
         Dbeta0N = zeros(length(tempN)-1,1);
         DbetaN = zeros(length(tempN)-1,size(data_raw,2));           
         for i=1:length(tempN)-1
            Dbeta0N(i) = ...
            get_Dbeta(ones(size(data_tempN,1),1),Fs_multiclassN,...
		     ParamN_old(tempN(i)).beta0,data_GTT_tempN,...
		     label_tempN,tempN(i),active_set_normal,...
		     num_active_normal,a_u,mode);
           temp_normal = temp_normal+Dbeta0N(i)^2;
           for j=1:size(data_raw,2)
             DbetaN(i,j) = ...
             get_Dbeta(data_tempN(:,j),Fs_multiclassN,...
		       ParamN_old(tempN(i)).beta(j),data_GTT_tempN,...
		       label_tempN,tempN(i),active_set_normal,...
		       num_active_normal,a_u,mode);
           end
           temp_normal = temp_normal+norm(DbetaN(i,:))^2;
         end
       end                              
       if count_step>1
         Dold = Dnew;
       end
    end

    %inter-normal update
    if num_active_normal>=2
        for i=1:length(tempN)-1
            ParamN_new(tempN(i)).beta0 = ...
            ParamN_old(tempN(i)).beta0-mu*Dbeta0N(i);%/norm(temp_normal);
            %unconstraint weights
            ParamN_new(tempN(i)).beta = ...
            ParamN_old(tempN(i)).beta-mu*DbetaN(i,:);%./norm(temp_normal);
        end
        Fs_multiclassN = ...
        getF_multiclass(data_tempN,data_GTT_tempN,label_tempN,...
        ParamN_new,active_set_normal);
        for i=1:length(tempN)-1
            L2 = L2 + ...
            norm([ParamN_new(tempN(i)).beta0 ParamN_new(tempN(i)).beta])^2;
        end
    end
   
    Dnew = ...
    ObjF_meta(Fs_multiclassN,L2,data_GTT_tempN,...
	      label_tempN,label,...
	      active_set_normal,a_u,mode);

    if Dnew<Dold
        updated = true;
        mu = step_size;
        if num_active_normal>=2
            ParamN_old = ParamN_new;
        end

    else
        updated = false;
        %reduce step size
        mu = 0.5*mu;
    end
end
end
