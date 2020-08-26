function omega = opt_lambda_cw(B, data_diff, lambda, V, J, T, U)
  Hess = B'*B;
  reg = lambda;
  yk = (Hess + reg.*eye(size(B,2)))\(norm(data_diff,2).*B'*T(:,1));
  foo = V(:,1:length(yk))*yk;
  omega = norm((J*foo - data_diff),2);
