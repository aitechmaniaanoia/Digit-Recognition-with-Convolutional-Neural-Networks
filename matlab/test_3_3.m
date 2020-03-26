%Q3.3
%% Network defintion
layers = get_lenet();

%% Loading data
xtest = [];
ytest = [];
D = '../test_data/';
S = dir(fullfile(D, '*.png'));

for img = 1:numel(S)
    F = fullfile(D,S(img).name);
    I = imread(F);
    I = rgb2gray(I);
    I = imresize(I,[28 28]);
    I = reshape(I,[784 1]);
    label = S(img).name(1:end-4);
    label = str2num(['uint8(',label,')']);
    
    xtest = [xtest I];
    ytest = [ytest label];
end
% load the trained weights
load lenet.mat

%% Testing
layers{1}.batch_size = 1;
for i=1:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i));
    test_label = ytest(i);
    [~,pred_label] = max(P);
    %pred_label = pred_label - 1;
    fprintf('Actual label is %d.\n', test_label);
    fprintf('Predict label is %d.\n', pred_label);
    %fprintf('prob: %.2f', a);
end

