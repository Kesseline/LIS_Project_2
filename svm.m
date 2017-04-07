function yfit = svm( XTRAIN, ytrain, XTEST, lambda )
%SVM Support vector machine

w = sgd(XTRAIN, ytrain, @svmgrad);

yfit = XTEST * w;

    function v = svmgrad(w, x, y)
        v = lambda*w;
        if (y * (x*w) < 1)
            b = x' * double(y);
            v = v - b;
        end
    end
end

