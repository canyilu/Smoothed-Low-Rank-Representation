function [Z, obj]= LRR_IRLS(X,lambda,rc,rho,display)

% Solve the smoothed LRR problem (6) by IRLS shown in the following paper:
% Canyi Lu, Zhouchen Lin, Shuicheng Yan, Smoothed Low Rank and Sparse
% Matrix Recovery by Iteratively Reweighted Least Squares Minimization,
% IEEE Transactions on Image Processing (TIP), 2014
% 
% Written by Canyi Lu (canyilu@gmail.com), December 2014.
%

if nargin < 5
    display = false;
end
[~, n] = size(X);
XtX = lambda*(X'*X);
maxiter = 500;
mu = rc*norm(X,2);
tol2 = 1e-5;
I = eye(n);
Z = zeros(n,n);
Z_old = Z;
W1 = eye(n,n);
W2 = ones(n,1); 
W = W1*diag(W2); 
if display
   obj = zeros(maxiter,1);
end
for t = 1 : maxiter
   % calculate Z: XtX*Z + Z*W - XtX = 0
   % X = lyap(A,B,C) solves AX+XB+C=0.
   Z = lyap(XtX,W,-XtX);
   
%  calculate W1 = (Z^T*Z+mu*I)^{-0.5} with SVD
%    [~,S,V]=svd(Z,'econ');  
%    s = diag(S);   
%    s = 1./sqrt( s.*s + mu^2 );
%    W1 = V*diag(s)*V';
   
   % or calculate W1 = (Z^T*Z+mu*I)^{-0.5} without SVD
   W1 = (Z'*Z+mu^2*I)^(-0.5);

   % calculate W2 which is a diagonal matrix
   E = X-X*Z;   
   E = dot(E,E);
   W2 = sqrt((E+mu^2));   
   W = W1*diag(W2);  
   
   % update mu
   mu = mu/rho; 
   
   % compute the objective function value
   if display
       EE = X-X*Z;
       obj(t) = nuclearnorm(Z)+lambda*sum(sqrt(sum(EE.*EE)));
   end   
   if norm(Z_old-Z,'fro')/norm(Z,'fro')<tol2       
       break;
   end 
   Z_old = Z;
end
if display
    if t<maxiter
       obj(t+1:end) = []; 
    end   
else    
    obj = [];
end

