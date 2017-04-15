%% Data preparation

%load data

load('../data/train.mat');
load('../data/xext.mat');
load('../data/test.mat');
load('../data/xtext.mat');
yrows = size(y,1);

% prepare data
Xn = X_ext';
yn = zeros(3,yrows);
for i = 1:yrows
    yn(y(i)+1,i) = 1;
end

%% Neural Net Creation

% Create a Pattern Recognition Network
hiddenLayerSize = 80;
net = patternnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;


% Train the Network
[net,tr] = trainlm(net,Xn,yn);

% Test the Network
outputs = net(Xn);
errors = gsubtract(yn,outputs);
performance = perform(net,yn,outputs)

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotconfusion(yn,outputs)
% figure, ploterrhist(errors)


%% Generate and export prediction with net
ypred = net(X_text');
ypred = round(ypred);
ypredt = zeros(3000,1);
for i = 1:3000
    if ypred(2,i) == 1
        ypredt(i) = 1;
    elseif ypred(3,i) == 1
            ypredt(i) = 2;
    end
end

export = table(Id,ypredt,'VariableNames',{'Id','y'});
writetable(export,'neural.csv');


%% Export Neural Net for further consideration
genFunction(net,'neuralnet');
