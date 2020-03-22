function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: 
h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

assert(h_out == floor(h_out), 'h_out is not integer');
assert(w_out == floor(w_out), 'w_out is not integer');
input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

%% Fill in the code
% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape.
%kernel = ones(k);
%find index of middle pixel in kernel
mid = (k + 1)/2;
%result = zeros([h_out, w_out, num]);
output.data = zeros([h_out, w_out, num, batch_size]);

for b = 1:batch_size
    %result = [];
    x = reshape(input.data(:,b),[h_in,w_in, c]);
    %x = padarray(x, [pad,pad], 0);
    [length,width,~] = size(x);
    for k_num = 1:num 
        %x = X(:,:,k_num);
        x1 = zeros(h_in,w_in);
        
        kernel = param.w(:,k_num);
        kernel = reshape(kernel,k,k,c);
        for i = 1:k
            for j = 1:k
                conv_img = kernel(i,j,:).*x;
                conv_img = sum(conv_img,3);
                if i > mid
                    %move top i-mid row to bottom
                    size_add = size(conv_img(1:(i-mid),:));
                    conv_img(1:(i-mid),:) = [];
                    conv_img = [conv_img; zeros(size_add)];
                elseif i < mid
                    %move bottom mid-i row to top
                    size_add = size(conv_img((length-mid+i+1):length,:));
                    conv_img((length-mid+i+1):length,:) = [];
                    conv_img = [zeros(size_add); conv_img];
                end

                if j > mid
                    % move left j-mid cloumn to right
                    size_add = size(conv_img(:,1:(j-mid)));
                    conv_img(:,1:(j-mid)) = [];
                    conv_img = [conv_img zeros(size_add)];
                elseif j < mid
                    % remove right mid-j cloumn
                    size_add = size(conv_img(:,(width-mid+j+1):width));
                    conv_img(:,(width-mid+j+1):width) = [];
                    conv_img = [zeros(size_add) conv_img];
                end

                % sum conv_img
                x1 = x1 + conv_img;
            end
        end
        %remove padding
        x1 = x1((mid-1+pad+1):(length-pad-mid+1), (mid-1+pad+1):(width-pad-mid+1));
        % x1 conv result for each img
        % add bias
        bias = param.b(k_num);
        x1 = x1 + bias;
        
        output.data(:,:,k_num,b) = x1; %[h_out, w_out, num, batch_size]
        %x1 = reshape(x1, [size(x1,1)*size(x1,2) 1]);
        %result = [result; x1];
    end
    %results = [results result];
end
output.data = reshape(output.data, [h_out*w_out*num,batch_size]);
%output.data = results;
output.height = h_out;
output.width = w_out;
output.channel = num;
output.batch_size = batch_size;
end

