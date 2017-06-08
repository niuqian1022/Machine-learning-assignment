function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%create a vector to have zero on the first element

J = -(log(1./(1+e.^(-X*theta)))'*y+log(1-1./(1+e.^(-X*theta)))'*(1-y))/m;
grad = X'*(1./(1+e.^(-X*theta))-y)/m;
w = theta;
w(1) = 0;
J = J + lambda*(w'*w)/(2*m);
grad = grad + lambda*w/m;





% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
