function [output] = relu_forward(input)
output.height = input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;

% Replace the following line with your implementation.
output.data = zeros(size(input.data));
% max(x,0);
for b = 1:input.batch_size
    data = input.data(:,b);
    data(data < 0) = 0;
    output.data(:,b) = data;
end
end
