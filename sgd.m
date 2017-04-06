function w = sgd( XTRAIN, ytrain, grad )
%SGD Stochastic gradient descent

[n,d] = size(XTRAIN);
w = zeros(d,1);
w_old = zeros(size(w));

t = 1;
while (norm(w - w_old) > 0.0000001 || t < 10)
    row = randi(n);
    w_old = w;
    w = w - (1/t)*grad(w, XTRAIN(row,:), ytrain(row));
    t = t + 1;
    delta = norm(w - w_old);
end

end

