function [dCostdWeight,dCostdBias] = BackProp(Input,Network,DesireOutput)
%BackProp.m
%  See http://neuralnetworksanddeeplearning.com/chap2.html for more
%   information on the back propagation algorithm.
%INPUT: Input - input vector to the neural network
%       Network - structure array representing the network
%          see the functions Network and Feedforward
%       DesireOutput - vector representing the correct output given the
%          input "Input" ... this is for the training/learning phase
%
%OUTPUT: dCostdWeight - cell array with matrices representing the partial 
%          derivative of the cost function with respect to the weights
%        dCostdBias - same as dCostdWeight, but for the biases
%
% Created: 2018/07/13, 24 Cummington, Boston
%  Byron Price
% Updated: 2018/07/16
%  By: Byron Price

[Output,Z] = Feedforward(Input,Network);

Activations = cell(1,Network.numCalcs);
Activations{1} = Input;
for ii=2:Network.numCalcs
    Activations{ii} = Output{ii-1};
end

dCostdWeight = cell(1,Network.numCalcs);
dCostdBias = cell(1,Network.numCalcs);

%deltaL = (Z{end}-DesireOutput); % linear output neuron, mean-squared error cost
deltaL = (Output{end}-DesireOutput); % cross-entropy cost with sigmoid/swish output neurons
                                % .*SwishPrime(Output{end}); % add this back for 
                                % mean-squared error cost function, unless
                                % you want the output neuron to be linear

for ll=Network.numLayers:-1:1
    activationIndex = (Network.numFilters+1)*(ll-1)+1;
    index = (Network.numFilters+1)*ll;
    temp = cat(3,Activations{activationIndex+1:activationIndex+Network.numFilters});
    dCostdWeight{index} = temp(:)*deltaL';
    dCostdBias{index} = deltaL;

    origDeltaL = deltaL;
    W = Network.Weights{index};
    fullSize = prod(Network.outputSize);
    for ii=1:Network.numFilters
        indeces = 1+fullSize*(ii-1):fullSize*ii;
        tmp = W(indeces,:)*origDeltaL;
        deltaL = kron(reshape(tmp,Network.outputSize),ones(Network.maxPool)).*SwishPrime(Z{activationIndex+ii-1});

        dCostdWeight{activationIndex+ii-1} = rot90(reshape(accumarray(Network.idx2,Activations{activationIndex}(Network.idx1).*deltaL(Network.idx3)),...
            [Network.networkStructure(2,1),Network.networkStructure(2,2)]),2);
        dCostdBias{activationIndex+ii-1} = sum(deltaL(:));
    end
%     if ll>1
%         divideIm = conv2(ones(Network.outputSize(ll,:)),ones(Network.networkStructure(2,:)));
%         newDeltaL = zeros(Network.outputSize(ll-1,:));
%         for ii=1:Network.numFilters
%             W = Network.Weights{index};
%             deltaL = (W(ii)*origDeltaL).*SwishPrime(Z{activationIndex+ii-1});
%             temp = conv2(deltaL,Network.Weights{activationIndex+ii-1});
%             temp = temp./divideIm;
%             newDeltaL = newDeltaL+temp;
%         end
%         deltaL = (newDeltaL./Network.numFilters).*SwishPrime(Z{(Network.numFilters+1)*(ll-1)});
%     end
end

end

% cross-entropy cost function
%  with neuron function a and desired output y
%  a might be the sigmoid function for example
% C = -1/n * SUM_x [yln(a)+(1-y)ln(1-a)]
%  in the output layer, we have
% dCostdWeight = 1/n * SUM_x [a(L-1)*(a(L)-y)]
% dCostdBias = 1/n * SUM_x [a(L)-y]
%  this corresponds to the exact same as above for the
%  quadratic cost function but with no multiplication
%  by sigmoid prime

%softmax cost function 
% outputs are exponentials that sum to 1
% cost is -ln(activation at desired output)
% dCostdWeight and dCostdBias as above
