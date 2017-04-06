
load('data/train.mat');

classif = @svm;
fun = @(XTRAIN,ytrain,XTEST)(onevone(XTRAIN,ytrain,XTEST,classif));

mcr = crossval('mcr',X,y,'Predfun',fun);

display(mcr);