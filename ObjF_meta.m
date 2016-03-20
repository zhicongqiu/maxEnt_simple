function D = ...
	 ObjF_meta(TARGETS,Fs_multiclassN,L2,...
		   data_GTT_N,label_N,label,...
		   active_set_normal,...
		   a_u,mode)

sumN = 0; sum2N = 0;

num_normal = sum(active_set_normal==1);
%inter-normal sum
if num_normal>=2
    %disp('calling...');
    temp = Fs_multiclassN(label_N~=2,:);
    data_GTT_N = data_GTT_N(label_N~=2,:);
    for i=1:size(temp,1)
        sumN = sumN - log(temp(i,data_GTT_N(i,1)));
    end
    Fn_U = Fs_multiclassN(label_N==2,:);
    tempN = find(active_set_normal==1);
end


if mode==2||mode==3
  if num_normal>=2
    for i=1:size(Fn_U,1)
      for j=1:length(tempN)
	temp_tar = TARGETS(i,tempN(j));
	if temp_tar~=0
	  sum2N = sum2N+...
		  temp_tar*(log(temp_tar)-log(Fn_U(i,tempN(j))));
	end        
      end
    end   
  end
end             

if mode~=1
  D = (1-a_u)*(sumN)+a_u*sum2N;
else
  D = (1-a_u)*(sumN)+a_u*L2;
end
       
