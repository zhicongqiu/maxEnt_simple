%generate three clusters
mu1 = [-6 0]; s1 = [2 1; 1 2];
mu2 = [0 0]; s2 = [2 1 ;1 2];
mu3 = [6 0]; s3 = [2 1 ;1 2];
%each 100 cases
d1 = [mvnrnd(mu1,s1,100) ones(100,1)];
d2 = [mvnrnd(mu2,s2,100) 2*ones(100,1)];
d3 = [mvnrnd(mu3,s3,100) 3*ones(100,1)];

%plot the data
%plot(d1(:,1),d1(:,2),'.k',d2(:,1),d2(:,2),'.k',d3(:,1),d3(:,2),'.k');

%label some samples, 10 from each
%get 10 highest from d1
[~,I_1] = sort(d1(:,2));
%get 10 lowest from d3
[~,I_3] = sort(d3(:,2));

DATA = [d1(I_1(end:-1:end-9),:);d3(I_3(1:10),:)...
	;d1(I_1(end-10:-1:1),:);d3(I_3(11:end),:);d2];
LABEL = DATA(:,3);
DATA = DATA(:,1:2);

%previous max ent approach
[ERR_maxEnt, SPA_maxEnt] = ...
semi_learning(DATA,LABEL,DATA(21:end,:),LABEL(21:end),[1 2 3],20,2);
%get a and b for a line plot
a_maxEnt = -SPA_maxEnt.ParamN(1).beta(1)/SPA_maxEnt.ParamN(1).beta(2);
b_maxEnt = -SPA_maxEnt.ParamN(1).beta0/SPA_maxEnt.ParamN(1).beta(2);
x = [-10:10];
y = a_maxEnt.*x+b_maxEnt.*x;
%plot(x,y,'-k','linewidth',2);
%

%joint learning
[ERR_joint,TARGETS,SPA_joint] = ...
semi_learning(DATA,LABEL,DATA(21:end,:),...
	      LABEL(21:end),[1 2 3],20,2);
%%alpha = 0.1
%jointly optimize z and beta
a_joint = -SPA_joint.ParamN(1).beta(1)/SPA_joint.ParamN(1).beta(2);
b_joint = -SPA_joint.ParamN(1).beta0/SPA_joint.ParamN(1).beta(2);
y_joint = a_joint.*x+b_joint.*x;
%
%{
%%alpha = 0.02
%jointly optimize z and beta
a_joint002 = -SPA_joint002.ParamN(1).beta(1)/SPA_joint002.ParamN(1).beta(2);
b_joint002 = -SPA_joint002.ParamN(1).beta0/SPA_joint002.ParamN(1).beta(2);
y_joint002 = a_joint002.*x+b_joint002.*x;
%
%%alpha = 0.2
%jointly optimize z and beta
a_joint02 = -SPA_joint02.ParamN(1).beta(1)/SPA_joint02.ParamN(1).beta(2);
b_joint02 = -SPA_joint02.ParamN(1).beta0/SPA_joint02.ParamN(1).beta(2);
y_joint02 = a_joint02.*x+b_joint02.*x;

%%
%plot as a function of alphas in joint model
plot(DATA(1:10,1),DATA(1:10,2),'or',...
     DATA(11:20,1),DATA(11:20,2),'sb',...
     x,y_joint02,':k',x,y_joint,'-k',x,y_joint002,'-.k','linewidth',2);
%}  
y_region = 2*x;
%plot the data
plot(d1(:,1),d1(:,2),'.k',d2(:,1),d2(:,2),'.k',d3(:,1),d3(:,2),'.k');
hold on;
H = plot(DATA(1:10,1),DATA(1:10,2),'or',...
	 DATA(11:20,1),DATA(11:20,2),'sb',...
	 x,y,'-k',x,y_joint,'-.k',x,y_region,':k','linewidth',2);
