function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% Transform y to Y (a 10*5000 matrix)
Y = zeros(m, num_labels);
for i = 1:m
Y(i,y(i)) = 1;
endfor

% Add ones to X (a1), then forward prop to compute a2 and a3
X = [ones(m, 1),X]; %5000 x 401
z2 = X*Theta1';    %5000 x 25
a2 = sigmoid(z2); %5000 x 25
a2 = [ones(m, 1), a2];  %5000 x 26
z3 = a2*Theta2';  %5000 x 10
a3 = sigmoid(z3);%5000 x 10 and Y is 5000 x 10
J = sum(sum(-(log(a3).*Y+log(1-a3).*(1-Y))))/m;

w1 = Theta1;
w2 = Theta2;
w1(:,1) = 0;
w2(:,1) = 0;

% with regularization
J = J + lambda*(sum(sum(w1.^2)) + sum(sum(w2.^2)))/(2*m);

% backpropapgation
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

d3 =a3 - Y; %5000 x 10
d2 = (d3*Theta2(:,2:end)).*sigmoidGradient(z2); 
         %5000 x 10, 10 x 25, 5000 x 25 -> 5000 x 25

Theta1_grad = Theta1_grad +(d2' * X)/m + lambda*w1/m; %Theta1_grad: 25.5000 x 5000.401
Theta2_grad = Theta2_grad +(d3' * a2)/m + lambda*w2/m;%Theta2_grad: 10.5000 x 5000.26
 
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
