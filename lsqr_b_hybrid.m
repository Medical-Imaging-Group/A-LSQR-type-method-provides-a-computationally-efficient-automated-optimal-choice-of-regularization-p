function [U,V,B,F] = lsqr_b_hybrid(A,b,k,reorth,s) 
%LSQR_B Solution of least squares problems by Lanczos bidiagonalization. 
% 
% [X,rho,eta,F] = lsqr_b(A,b,k,reorth,s) 
% 
% Performs k steps of the LSQR Lanczos bidiagonalization algorithm 
% applied to the system 
%    min || A x - b || . 
% The routine returns all k solutions, stored as columns of 
% the matrix X.  The solution norm and residual norm are returned 
% in eta and rho, respectively. 
% 
% If the singular values s are also provided, lsqr computes the 
% filter factors associated with each step and stores them columnwise 
% in the matrix F. 
% 
% Reorthogonalization is controlled by means of reorth: 
%    reorth = 0 : no reorthogonalization (default), 
%    reorth = 1 : reorthogonalization by means of MGS.
 
% Reference: C. C. Paige & M. A. Saunders, "LSQR: an algorithm for 
% sparse linear equations and sparse least squares", ACM Trans. 
% Math. Software 8 (1982), 43-71. 
 
% Per Christian Hansen, IMM, April 8, 2001. 
 
% The fudge threshold is used to prevent filter factors from exploding. 
fudge_thr = 1e-4; 
 
% Initialization. 
if (k < 1), error('Number of steps k must be positive'), end 
if (nargin==3), reorth = 0; end 
if (nargout==4 & nargin<5), error('Too few input arguments'), end 
[m,n] = size(A); X = zeros(n,k); 
if (reorth==0) 
  UV = 0; 
elseif (reorth==1) 
  U = zeros(m,k); V = zeros(n,k); UV = 1; 
  if (k>=n), error('No. of iterations must satisfy k < n'), end 
else 
  error('Illegal reorth') 
end 
 
% Prepare for LSQR iteration. 
v = zeros(n,1); x = v; beta = norm(b);  
if (beta==0), error('Right-hand side must be nonzero'), end 
if (reorth==2) 
  [beta,HHbeta(1),HHU(:,1)] = gen_hh(b); 
end 
u = b/beta; if (UV), U(:,1) = u; end 
r = (u'*A)'; alpha = norm(r);   % A'*u; 
v = r/alpha; if (UV), V(:,1) = v; end 
phi_bar = beta; rho_bar = alpha; w = v; 
if (nargin==5), Fv = s/(alpha*beta); Fw = Fv; end 
 B = zeros(k+1,k);
% Perform Lanczos bidiagonalization with/without reorthogonalization. 
for i=2:k+1 
  
  alpha_old = alpha; beta_old = beta; 
 
  % Compute A*v - alpha*u. 
  p = A*v - alpha*u; 
  if (reorth==0) 
    beta = norm(p); u = p/beta; 
  else
    for j=1:i-1, p = p - (U(:,j)'*p)*U(:,j); end 
    beta = norm(p); u = p/beta; 
  end 
 
  % Compute A'*u - beta*v. 
  r = (u'*A)' - beta*v;   % A'*u 
  if (reorth==0) 
    alpha = norm(r); v = r/alpha; 
  else
    for j=1:i-1, r = r - (V(:,j)'*r)*V(:,j); end 
    alpha = norm(r); v = r/alpha; 
  end 
 
  % Store U and V if necessary. 
  if (UV), U(:,i) = u; V(:,i) = v; end 
 
  % Construct and apply orthogonal transformation. 
  rrho = pythag(rho_bar,beta); c1 = rho_bar/rrho; 
  s1 = beta/rrho; theta = s1*alpha; rho_bar = -c1*alpha; 
  phi = c1*phi_bar; phi_bar = s1*phi_bar; 
 
w = v - (theta/rrho)*w; 

 B(i-1,i-1) = alpha_old;
  B(i,i-1) = beta;
end 
B(i,i) = alpha;

