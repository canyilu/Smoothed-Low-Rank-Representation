
% IRLS for LRR problem

clear;
close all;
addpath(genpath(cd));

fprintf('\n\n**************************************   %s   *************************************\n' , datestr(now) );


% exp 1
k = 15;    % number of subspaces
n = 20;     % samples in each subspace
d = 200;    % dimension
r = 5;      % rank

[X,gnd] = generate_data(n,r,d,k);
[d, n]=size(X);
lambda = 0.1;  

%% IRLS
tic
[Z objs] = LRR_IRLS(X,lambda,0.1,1.1,1);
time = toc;
E = X-X*Z;
obj = nuclearnorm(Z)+lambda*sum(sqrt(sum(E.*E)));
iter = length(objs);
plot(objs)

fprintf('Minimum \t Time \t Iter.\n' );
fprintf('%f \t %f \t %d\n', obj, time, iter);
