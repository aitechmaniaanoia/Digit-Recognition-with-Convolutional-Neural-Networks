function [output] = pooling_layer_forward(input, layer)
% max pooling
    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    output.data = zeros([h_out, w_out, c, batch_size]);
    for b = 1:batch_size
        data = input.data(:,b);
        data = reshape(data, [h_in,w_in,c]);
        %data = padarray(data, [pad,pad],0);
        for i = 1:h_out
            for j = 1:w_out
                num = 0;
                kernel = data(i+k*num:i+(k*num+1), j+k*num:j+(k*num+1),:);
                output.data(i,j,:,b) = max(kernel,[],[1 2]);
                num = num + 1;
            end
        end
    end
    output.data = reshape(output.data, h_out*w_out*c, batch_size);
end

