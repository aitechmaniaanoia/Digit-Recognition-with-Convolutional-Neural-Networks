function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

% Replace the following lines with your implementation.
batch_size = input.batch_size;
%input_od = zeros(size(input.data));
param_grad.b = zeros(size(param.b));
param_grad.w = zeros(size(param.w));

input_od = param.w*output.diff;
param_grad.w = input.data*output.diff';
param_grad.b = ones([1 batch_size])*output.diff';
end
