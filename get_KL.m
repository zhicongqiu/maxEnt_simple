function KL = get_KL(target,I,post)

  KL = 0;
  for i=1:length(I)
    if target(I(i))~=0
       if post(I(i))==0
	  KL = Inf;
	  return ;
       else
	 KL = KL+target(I(i))*log(target(I(i))/post(I(i)));
       end
    end
  end
