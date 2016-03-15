function [error error_avgC]= ...
	 calculate_error(test_PMFnormal,...
			 test_GTT,count)

%over-all classification error
M = size(test_PMFnormal,1);
total_normal = sum(test_GTT(:,2)==0);

error = 0;
error_avgC = 0;

error_normalPC= zeros(1,size(test_PMFnormal,2));
total_normalPC = zeros(1,size(test_PMFnormal,2));

for i=1:M
  total_normalPC(test_GTT(i,1)) = total_normalPC(test_GTT(i,1))+1;
  [an bn] = max(test_PMFnormal(i,:));
  if test_GTT(i,1)~=bn %inter-normal class error
    error = error+1;
    error_normalPC(test_GTT(i,1)) = error_normalPC(test_GTT(i,1))+1;
  end     
end

if count == 0 %report error rate on test set
    if M~=0
        error = error/M;
    else
        error = 0;
    end    
    error_avgC = ...
    mean(error_normalPC(total_normalPC~=0)./total_normalPC(total_normalPC~=0));
end

end
