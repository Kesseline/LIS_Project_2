function yfit = onevone ( XTRAIN, ytrain, XTEST, classif )
% ONEVONE One vs One scheme for multiclass classification

[n,~] = size(ytrain);
[m,~] = size(XTEST);

%% Convert ytrain from [nominal] to [uint32] y

y = uint32(ytrain);
k = max(y);

%% Convert y to y_mod
% Convert y to nxk matrix y_mod where y_mod(i,j) == 1 <=> y(i) = j and
% y_mod(i,j) == -1 <=> y(i) != j

y_mod = int32(zeros(n,3));
for i=1:n
    y_mod(i, y(i)) = 1;
end

%% Compute RES
% For each distinct pair (i,j) of classes:
% Filter [ y_mod XTRAIN ] to include only rows that belong to i or j.
% Compute RES(:,i,j) = XTEST * w_ij where w_ij is the support vector returned by
% the classifier classif.

RES = zeros(m,k,k);

%trained lambda seems overfitted
%lambda = [0    0.6000    0.3235;
%     0.6000         0    0.0515;
%     0.3235    0.0515         0];

for i=1:k
    for j=(i+1):k
        TEMP = [ double(y_mod) XTRAIN ];
        indI = TEMP(:,i) == 1;
        indJ = TEMP(:,j) == 1;
        filt = indI | indJ;
        yX   = TEMP(filt, :);
        y    = int32(yX(:,1:3));
        X    = yX(:,4:end);
        RES(:,i,j) = classif( X, (y(:,i)-y(:,j)), XTEST i, j);
    end
end

%% Count
% For each input data point count the number of victories for each class.

SUM = zeros(m,k);

for i=1:k
    for j=(i+1):k
        for a=1:m
            if (RES(a,i,j) > 0)
                SUM(a,i) = SUM(a,i) + 1;
            else
                SUM(a,j) = SUM(a,j) + 1;
            end
        end
    end
end

%% Vote
% Select the class with most victories and write class to yfit

yfit = zeros(m,1);

for i=1:m
    temp = zeros(2,1);
    for j=1:k
        if (SUM(i,j) > temp(1))
            temp = [SUM(i,j) j]';
        end
    end
    yfit(i) = temp(2)-1;
end

end