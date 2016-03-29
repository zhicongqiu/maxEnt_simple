function Dbeta = get_Dbeta(data,TARGETS,Fs_multiclass,w,data_GTT_temp,...
			   label_temp,l,active_set,WT,w_u,num,a_u,mode)

sum0 = 0;
sum1 = 0;
sum2 = 0;
count = 0;
for i=1:size(Fs_multiclass,1)
    if label_temp(i)~=2 %labeled class
        if data_GTT_temp(i,1)==l %same-class label
            sum0 = sum0-...
		   WT(l)*(1-Fs_multiclass(i,l))*data(i);
        else %diff-class label
            sum1 = sum1+...
		   WT(data_GTT_temp(i,1))*Fs_multiclass(i,l)*data(i);
        end
    else %unknown class
      count = count+1;
      if mode==2||mode==3
        temp = find(active_set==1);
        for j=1:length(temp)
	  %check if the mass is non-zero for class j
	  if TARGETS(count,j)~=0
            if temp(j)==l
              sum2 = sum2-...
		     TARGETS(count,temp(j))*(1-Fs_multiclass(i,l))*data(i);
            else
              sum2 = sum2+...
		     TARGETS(count,temp(j))*Fs_multiclass(i,l)*data(i);
            end
	  end
        end
      end
    end
end

if mode~=1
    Dbeta = (1-a_u)*(sum0+sum1)+a_u*w_u*sum2;
else
    Dbeta = (1-a_u)*(sum0+sum1)+a_u*2*w;
end
end
