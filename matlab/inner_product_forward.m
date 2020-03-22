function [output] = inner_product_forward(input, layer, param)
% fully connected layer
% layer.type = 'IP';
% layer.num = 500;
% layer.init_type = 'uniform';

d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);

% Replace the following line with your implementation.
% w [m n] m this layer n previous layer
% b [m 1]
output.data = zeros([n, k]);
output.data = w*input + b;

end
