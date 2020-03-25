function [output] = inner_product_forward(input, layer, param)
% fully connected layer
% layer.type = 'IP';
% layer.num = 500;
% layer.init_type = 'uniform';

d = size(input.data, 1);
k = size(input.data, 2); % batch size
%n = size(param.w, 2);
n = layer.num;

% Replace the following line with your implementation.
% w [m n] m this layer n previous layer
% b [m 1]
output.data = zeros([n, k]);
val = param.w'*input.data;
for b = 1:k
    output.data(:,b) = val(:,b) + param.b';
end

output.height = n;
output.width = 1;
output.channel = 1;
output.batch_size = k;
end
