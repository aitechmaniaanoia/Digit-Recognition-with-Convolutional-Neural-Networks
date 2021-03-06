% Q5
%% Network defintion
layers = get_lenet();

%% Loading data
xtest = [];
D = '../images/';
S = dir(fullfile(D, '*.jpg'));
train_img = 1;

for img = 1:numel(S)
    F = fullfile(D,S(img).name);
    I = imread(F);
    I = rgb2gray(I);
    I(I>100) = 0;
    I(I>0) = 255;
    
    %%Remove all object containing fewer than 30 pixels
    I = bwareaopen(I,30);
    % extract single digit
    [L,~]=bwlabel(I);
    %%Measure properties of image regions
    propied = regionprops(L, 'BoundingBox');
    
    % extract digit from bounding box
    for n=1:size(propied,1)
        %rectangle('Position',propied(n).BoundingBox,'EdgeColor','r','LineWidth',1);
        x1 = int16(propied(n).BoundingBox(2));
        y1 = int16(propied(n).BoundingBox(1));
        
        x2 = int16(x1 + propied(n).BoundingBox(4));
        y2 = int16(y1 + propied(n).BoundingBox(3));
        data = I(x1:x2, y1:y2);
        data = padarray(data, [5,5], 0);
        data = imresize(data,[28 28]);
        % save image 
        filename = fullfile('../results', [int2str(train_img), '.jpg']);
        imwrite(data, filename);
        train_img = train_img + 1;
        
        data = reshape(data,[784 1]);
        xtest = [xtest data];
    end
end

% load the trained weights
load lenet.mat

%% Testing
pred_labels = [];
layers{1}.batch_size = 1;
for i=1:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i));
    [~,pred_label] = max(P);
    pred_label = pred_label - 1;
    pred_labels = [pred_labels pred_label];
end

