function [Output,Z] = Feedforward(Input,Network)
%Feedforward.m
% This code will take as input a convolutional network model and
%   also a series of inputs to its first layer.  It
%   will then calculate a feedforward output from the network assuming
%   swish neurons at each of the network's nodes.
%INPUT: Input - matrix of inputs to the network
%       Network - structure array representing our
%          Network, see the function Network
%
%OUTPUT: Output - cell array of the outputs from each layer of the network,
%           starting with the second layer (the first layer is purely an
%           input layer).
%        Z - cell array of the weighted and biased inputs to each layer,
%           starting with the second layer, these are often called
%           activations
% 
% Created: 2018/07/13, 24 Cummington, Boston
%  Byron Price
% Updated: 2018/07/17
%  By: Byron Price

Output = cell(1,Network.numCalcs);
Z = cell(1,Network.numCalcs);

X = Input;
for jj=1:Network.numLayers
    filterIndex = Network.numFilters(jj)*(jj-1)+1;
    idx = kron(reshape(1:(Network.outputSize{jj}(1)*Network.outputSize{jj}(2)),Network.outputSize{jj}),ones(Network.maxPool(jj)));
    OutputMatrix = zeros([Network.outputSize{jj},Network.numFilters(jj)]);
    for ii=1:Network.numFilters(jj)
        if jj==1
            temp = conv2(X,Network.Weights{filterIndex},'valid');
        else
            temp = conv2(squeeze(X(:,:,1)),Network.Weights{filterIndex},'valid');
            for kk=2:Network.numFilters(jj-1)
                temp = temp+conv2(squeeze(X(:,:,kk)),Network.Weights{filterIndex},'valid');
            end
%             temp = conv2(squeeze(X(:,:,ii)),Network.Weights{filterIndex},'valid');
        end
        Z{filterIndex} = temp+Network.Biases{filterIndex};
        temp = Swish(Z{filterIndex});
        temp = reshape(accumarray(idx(:),temp(:),[],@max),Network.outputSize{jj});
        Output{filterIndex} = temp;
        OutputMatrix(:,:,ii) = temp;
        filterIndex = filterIndex+1;
    end
    X = OutputMatrix;
%     Z{index} = sum(OutputMatrix,3)+Network.Biases{index};
%     Output{index} = Swish(Z{index});
%     X = Output{index};
end

Z{filterIndex} = Network.Weights{filterIndex}'*X(:)+Network.Biases{filterIndex};
Output{filterIndex} = Swish(Z{filterIndex});

end
