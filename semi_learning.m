function [ERROR, SPARSITY] = semi_learning(data_raw,data_GT,...
						  test_raw,test_GT,...
						  normal_class,num_train,mode)
%unction [ERROR SPARSITY] = ...
%        semi_learning(data_raw,data_GT,...
%		       test_raw,test_GT,...
%		       normal_class,...
%		       num_train,mode)
%input: 
%training data, N samples and K features in raw space, labels, 
%test data in raw space, labels
%first num_train samples are used as labeled samples
%mode:
%0:MLE regularization 
%1:L2 regularization 
%2: proposed maxEnt regularizer 
%3: min_ent regularizer without L2 (-1*maxEnt), not to be confused
%with Bengio's min ent reg
%output:
%ERROR metrics

%make sure there is the specified mode
if ~ismember(mode,[0 1 2 3])
   error('please specify modes: 0, 1, 2 or 3')
end
%make sure train and test has the same input dimension
if size(data_raw,2)~=size(test_raw,2)
   error('train and test dimension mismatch')
end
%make sure number of samples is the same as number of labels
if length(data_raw)~=length(data_GT)
   error('number of training samples should be equal to number of labels')
end
if length(test_raw)~=length(test_GT)
   error('number of test samples should be equal to number of labels')
end


[N, K_O] = size(data_raw);
if num_train>N
   num_train = N; %at most all labeled
end

label = 2*ones(N,1); %2 is unlabeled, 0 is labeled normal


%ground-truth transformation, 1st column indicates its Param index; 
%2nd column indicates if it is normal (0)
data_GTT = transform_GT(data_GT,normal_class);
test_GTT = transform_GT(test_GT,normal_class);

%initialize parameters
active_set_normal = zeros(length(normal_class),1);

for i=1:num_train
  if active_set_normal(data_GTT(i,1))==0
    active_set_normal(data_GTT(i,1)) = 1;
  end
end
label(1:num_train) = 0;
if sum(active_set_normal)<2
   fprintf('there is one class labeled in learning...\n')
end

SPARSITY = struct;
ERROR = struct;
ParamN = struct;
for i=1:length(normal_class)    
  ParamN(i).beta = 1e-6*ones(1,K_O);
  ParamN(i).beta0 = 0;
end
    
a_u = 0.2;
%{
a_u = DoCV(data_raw,K_O,data_GTT,label,...
	   active_set_normal,true,...
	   mode);
%}
disp(a_u);


%initialize targets
%TARGETS


%0:MLE regularization or 1:L2 regularization or 2:proposed model or 3: min_ent 
[ParamN] = ...
gradDes(data_raw,data_GTT,label,a_u,ParamN,...
	active_set_normal,mode);

%test PMF based on current parameters
[test_PMFnormal] = ...
getTestPMF(test_raw,test_GTT,ParamN,...
	   active_set_normal);
    
%error rate on test set
[classification_error, error_avgC] = ...
calculate_error(test_PMFnormal,test_GTT,0);
disp(error_avgC);

SPARSITY.ParamN = ParamN(active_set_normal==1);
SPARSITY.active_set_normal = active_set_normal;
ERROR.error = classification_error;
ERROR.avgC = error_avgC;    
