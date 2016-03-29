function [count_step ParamN_old,TARGETS] = ...
         gradDes(data_raw,data_GTT,label,a_u,...
		 ParamN_old,active_set_normal,WT,w_u,mode)

%initial step size and tolerance level
step_size = 1e-2;
reltol = 1e-4;

%initialize parameters
num_active_normal = sum(active_set_normal);
%data_GTT_tempN = 0;
%data_tempN = 0;
%label_tempN = 0;
L2 = 0;
ParamN_new = ParamN_old;
tempN = find(active_set_normal==1);
%{
if num_active_normal>=2
    %filter out rare samples
    data_tempN = data_raw(label~=1,:);
    data_GTT_tempN = data_GTT(label~=1,:);
    label_tempN = label(label~=1);
end
%}
Fs_multiclassN = 0;

Fs_multiclassN = ...
getF_multiclass(data_raw,data_GTT,label,...
		ParamN_old,active_set_normal);
for i=1:length(tempN)-1
  L2 = L2 + ...
       norm([ParamN_old(tempN(i)).beta0 ParamN_old(tempN(i)).beta])^2;
end

%initialize target distribution
num_unlabeled = length(label(label~=0));
TARGETS = 1/num_active_normal*ones(num_unlabeled,length(active_set_normal));
%calculate initial objective function value
Dold =  ObjF_meta(TARGETS,Fs_multiclassN,L2,...
		  data_GTT,label,active_set_normal,...
		  WT,w_u,a_u,mode);
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
       Fs_multiclassN = ...
       getF_multiclass(data_raw,data_GTT,label,...
		       ParamN_old,active_set_normal);
    
       %update parameters for each class           
       %initialize Dbeta0 and Dbeta
       Dbeta0N = zeros(length(tempN)-1,1);
       DbetaN = zeros(length(tempN)-1,size(data_raw,2));           
       for i=1:length(tempN)-1
         Dbeta0N(i) = ...
         get_Dbeta(ones(size(data_raw,1),1),TARGETS,Fs_multiclassN,...
		   ParamN_old(tempN(i)).beta0,data_GTT,...
		   label,tempN(i),active_set_normal,...
		   WT,w_u,num_active_normal,a_u,mode);
         temp_normal = temp_normal+Dbeta0N(i)^2;
         for j=1:size(data_raw,2)
           DbetaN(i,j) = ...
           get_Dbeta(data_raw(:,j),TARGETS,Fs_multiclassN,...
		     ParamN_old(tempN(i)).beta(j),data_GTT,...
		     label,tempN(i),active_set_normal,...
		     WT,w_u,num_active_normal,a_u,mode);
         end
         temp_normal = temp_normal+norm(DbetaN(i,:))^2;
       end                            
       if count_step>1
         Dold = Dnew;
       end
    end

    %inter-normal update
    for i=1:length(tempN)-1
      ParamN_new(tempN(i)).beta0 = ...
      ParamN_old(tempN(i)).beta0-mu*Dbeta0N(i);%/norm(temp_normal);
      %unconstraint weights
      ParamN_new(tempN(i)).beta = ...
      ParamN_old(tempN(i)).beta-mu*DbetaN(i,:);%./norm(temp_normal);
    end
    Fs_multiclassN = ...
    getF_multiclass(data_raw,data_GTT,label,...
		    ParamN_new,active_set_normal);
    for i=1:length(tempN)-1
      L2 = L2 + ...
           norm([ParamN_new(tempN(i)).beta0 ParamN_new(tempN(i)).beta])^2;
    end
   
    Dnew = ...
    ObjF_meta(TARGETS,Fs_multiclassN,L2,data_GTT,...
	      label,active_set_normal,WT,w_u,a_u,mode);

    if Dnew<Dold
        updated = true;
	fprintf('update parameters...\n');
	%reset step size
        mu = step_size;
        ParamN_old = ParamN_new;
	%%
	if mode ==2
	  %update targets
	  Fs_multiclassU = Fs_multiclassN(label==2,:);
	  data_GTT_tempU = data_GTT(label==2,:);
	  for i=1:size(Fs_multiclassU,1)
	    lowest_KL = Inf;
	    lowest_target = 1/num_active_normal...
			    *ones(1,length(active_set_normal));
	    [~,temp_I] = sort(Fs_multiclassU(i,:),'descend');
	    for j=1:num_active_normal
	      temp_target = zeros(1,length(active_set_normal));
	      %assign uniform target
	      for k=1:j
		temp_target(temp_I(k)) = 1/j;
	      end
	      temp_KL = get_KL(temp_target,temp_I(1:j),Fs_multiclassU(i,:));
	      %fprintf('tempKL is %f \n',temp_KL);
	      %get the lowest KL target
	      if temp_KL<lowest_KL
		%disp('using the loop...');
		lowest_KL = temp_KL;
		lowest_target = temp_target;
	      end
	    end
	    %assign the target distribution which reduces the objective
	    %the most
	    TARGETS(i,:) = lowest_target;
	  end
	end
	%%
    else
        updated = false;
        %reduce step size
        mu = 0.5*mu;
    end
end
end
