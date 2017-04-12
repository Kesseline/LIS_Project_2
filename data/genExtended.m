load('train.mat');

% Generate min, max, avg and median as additional features

x_min = min(X,[],2);
x_max = max(X,[],2);
x_avg = mean(X,2);
x_med = median(X,2);

X_ext = [X, x_min, x_max, x_avg, x_med];

save('xext.mat', 'X_ext');

% Experiments in Classification App show that it gets approx. the same
% results as before in terms of percision. However the models used are much
% simpler (ie weighted KNN, quadratic SVM)