function [PMFnormal] = ...
	 getTestPMF(test_raw,test_GTT,ParamN,...
		    active_set_normal)

M = size(test_raw,1);

if sum(active_set_normal)>=2
  PMFnormal = getF_multiclass(test_raw,test_GTT,2*ones(M,1),...
			      ParamN,active_set_normal);
else
    PMFnormal = zeros(M,length(active_set_normal));
    PMFnormal(:,active_set_normal==1) = 1;
end

end
    
