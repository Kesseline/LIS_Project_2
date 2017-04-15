function w = sgd( XTRAIN, ytrain, grad )
%SGD Stochastic gradient descent
% Performs stochastic gradient descent over the gradient grad.
% Termination condition is norm(w_(i) - w_(i-1)) < ep

ep = 0.0000001;

[n,d] = size(XTRAIN);

w = zeros(d,1);
w_old = zeros(size(w));

t = 1;
while (norm(w - w_old) > ep || t < 10)
    row = randi(n);
    w_old = w;
    w = w - (1/t)*grad(w, XTRAIN(row,:), ytrain(row));
    t = t + 1;
end

end

