function [input_od] = relu_backward(output, input, layer)

batch_size = input.batch_size;
% Replace the following line with your implementation.
input_od = zeros(size(input.data));
for b = 1:batch_size
    h_1 = input.data(:,b);
    %h = output.data(:,b);
    grad = zeros(size(h_1));
    grad(h_1>=0) = 1;
    input_od(:,b) = output.diff(:,b).*grad;
end
end
