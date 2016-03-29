function [a_u] = DoCV(data_raw,N_O,data_GTT,...
		      label,active_set_normal,...
		      reinitialize,mode)

start = 0.9;
ParamN = struct;

%10-fold CV
nr_fold = 10;

if sum(active_set_normal)==0 %no suspicious sample, choose a_u=0.1
    a_u = 0.1;
    return;
else
    %we have label_train_set,label_test_set and unlabeled samples
    train_raw = data_raw(label~=2,:);
    train_GTT = data_GTT(label~=2,:);
    train_label = label(label~=2);
    train_M = size(train_raw,1);
    
    unlabeled_raw = data_raw(label==2,:);
    unlabeled_GTT = data_GTT(label==2,:);
    unlabeled_label = label(label==2);
    
    %create 10-fold
    rand_index = randsample(train_M,train_M);
    fold_start(1)=0; %for numerical reason
    for i=1:nr_fold
        temp = round(i*train_M/nr_fold);
        fold_start(i+1) = temp;
    end

    temp_error = 1;
    for i=start:-0.1:0.1
        %reinitialize parameter?
        if reinitialize==true||i==start
            for k=1:length(active_set_normal)    
              ParamN(k).beta = 1e-6*ones(1,N_O);
              ParamN(k).beta0 = 0;
            end
        end
        
        total_error = 0;

        %10-fold CV
        for j=1:nr_fold
          %1-fold for testing
          testCV_raw = ...
	  train_raw(rand_index(fold_start(j)+1:fold_start(j+1)),:);
          testCV_GTT = ...
	  train_GTT(rand_index(fold_start(j)+1:fold_start(j+1)),:);
          testCV_label = ...
	  train_label(rand_index(fold_start(j)+1:fold_start(j+1)),:);
            
          %the other folds for training
          trainCV_raw = ...
          [train_raw([rand_index(1:fold_start(j));rand_index(fold_start(j+1)+1:end)],:);unlabeled_raw];
          trainCV_GTT = ...
          [train_GTT([rand_index(1:fold_start(j));rand_index(fold_start(j+1)+1:end)],:);unlabeled_GTT];
          trainCV_label = ...
          [train_label([rand_index(1:fold_start(j));rand_index(fold_start(j+1)+1:end)],:);unlabeled_label];

          %active set redefined
          active_set_normalT = zeros(length(active_set_normal),1);
          if sum(active_set_normal)>=2               
            for k=1:size(trainCV_raw,1)
              if trainCV_label(k)==0&&active_set_normalT(trainCV_GTT(k,1))==0
                active_set_normalT(trainCV_GTT(k,1)) = 1;
              end
            end
          end

          %gradient descend for training fold
          [ParamN] = ...
          gradDes(trainCV_raw,trainCV_GTT,trainCV_label,...
		  i,ParamN,active_set_normalT,mode);
          %use classification error, unweighted or weighted???
          [test_PMFnormal] = ...
	  getTestPMF(testCV_raw,testCV_GTT,...
		     ParamN,active_set_normalT);  
          [error error_avgC] = ...
	  calculate_error(test_PMFnormal,...
			  testCV_GTT,0);

	  %use weighted error as criterion
	  total_error = total_error+error_avgC;
        end
	
	%averaged over all folds
	weighted_error = total_error/nr_fold;

        %disp(weighted_error);       
        if weighted_error <= temp_error
          temp_error = weighted_error;
          a_u = i;            
        end
        
    end
    
end
