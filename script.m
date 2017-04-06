
load('data/train.mat');

fun = @(XTRAIN,ytrain,XTEST)(onevone(XTRAIN,ytrain,XTEST,@svm));

mcr = crossval('mcr',X,y,'Predfun',fun);

display(mcr);