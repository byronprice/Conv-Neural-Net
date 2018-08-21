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
    filterIndex = (Network.numFilters+1)*(jj-1)+1;
    index = (Network.numFilters+1)*jj;
    OutputMatrix = zeros(Network.outputSize(jj,:));
    for ii=1:Network.numFilters
        temp = conv2(X,Network.Weights{filterIndex},'valid');
        Z{filterIndex} = temp+Network.Biases{filterIndex};
        Output{filterIndex} = Swish(Z{filterIndex});
        OutputMatrix = OutputMatrix+Output{filterIndex}.*Network.Weights{index}(ii);
        filterIndex = filterIndex+1;
    end
    
    Z{index} = OutputMatrix+Network.Biases{index};
    Output{index} = Swish(Z{index});
    X = Output{index};
end

end
