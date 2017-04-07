function a = kernperceptron( XTRAIN, ytrain, XTEST, kern )
%KERNPERCEPTRON Summary of this function goes here
%   Detailed explanation goes here

[n,~] = size(ytrain);
a = zeros(n,1);

t = 1;
while (t < 10000)
    i = randi(n);
    predict = 0;
    for j=1:n
        predict = predict + a(j)*ytrain(j)*kern(XTRAIN(j,:)', XTRAIN(i,:)');
    end
    if (predict * ytrain(i) <= 0)
        a(i) = a(i) + (1/t);
    end
    t = t+1;
end

yfit = zeros(m,1);

for i=1:m
    yfit(i) = 0;
    for j=1:n
        yfit(i) = yfit(i) + a(j)*y_alt(j)*kern(XTRAIN(j,:)', XTEST(i,:)');
    end
end

end

