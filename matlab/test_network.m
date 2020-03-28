%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix
confusion_m = zeros(10);
correct = 0;

for i=1:100:size(xtest, 2)
    pred_labels = [];
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    test_label = ytest(:, i:i+99);
    for j = 1:size(P,2)
        [prob, pred_label] = max(P(:,j));
        confusion_m(test_label(j), pred_label) = confusion_m(test_label(j), pred_label) + 1;
        pred_labels = [pred_labels pred_label];
    end

    % calculate accuracy
    correct = correct + sum(pred_labels == test_label);
end
accuracy = correct/size(xtest, 2);
%confusion_m = confusion_m/size(xtest, 2);

