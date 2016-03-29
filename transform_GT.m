function data_GTT = transform_GT(GT,normal_class)

data_GTT = zeros(length(GT),2);
for i=1:length(GT)
    temp = find(GT(i)==normal_class);
    if isempty(temp)
      disp(GT(i));
      error('a sample has no ground-truth?');
    else
      data_GTT(i,1)=temp;
      data_GTT(i,2) = 0;
    end
end
