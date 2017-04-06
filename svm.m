function w = svm( XTRAIN, ytrain )
%SVM Support vector machine

lambda = 1;
w = sgd(XTRAIN, ytrain, @svmgrad);

    function v = svmgrad(w, x, y)
        v = lambda*w;
        if (y * (x*w) < 1)
            b = x' * double(y);
            v = v - b;
        end
    end
end

