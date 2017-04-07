
load('data/train.mat');

fun = @(XTRAIN,ytrain,XTEST)(onevone(XTRAIN,ytrain,XTEST,@classifier));

mcr = crossval('mcr',X,y,'Predfun',fun);

display(mcr);

%% Classifier function
function yfit = classifier(XTRAIN, ytrain, XTEST, class_i, class_j)
    % For each to classes a different classifier can be chosen.
    if ((class_i == 1 && class_j == 2) || (class_i == 2 && class_j == 1))
        yfit = svm(XTRAIN,ytrain,XTEST,0.6);
    elseif ((class_i == 1 && class_j == 3) || (class_i == 3 && class_j == 1))
        yfit = svm(XTRAIN,ytrain,XTEST,1);
    elseif ((class_i == 2 && class_j == 3) || (class_i == 3 && class_j == 2))
        yfit = svm(XTRAIN,ytrain,XTEST,1);
    else
        yfit = svm(XTRAIN,ytrain,XTEST,1);
    end
end