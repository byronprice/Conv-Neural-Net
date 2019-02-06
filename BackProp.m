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
activationIndex = Network.numFilters(end)*(Network.numLayers-1)+1;
index = Network.numCalcs;
temp = cat(3,Activations{activationIndex+1:activationIndex+Network.numFilters(end)});
dCostdWeight{index} = temp(:)*deltaL';
dCostdBias{index} = deltaL;

origDeltaL = deltaL;
W = Network.Weights{index};
fullSize = prod(Network.outputSize{end});

activationIndex = Network.numFilters(end)*(Network.numLayers-2)+1;
neuronIndex = Network.numFilters(end)*(Network.numLayers-1)+1;
for ii=1:Network.numFilters(end)
    indices = 1+fullSize*(ii-1):fullSize*ii;
    tmp = W(indices,:)*origDeltaL;
    deltaL = kron(reshape(tmp,Network.outputSize{end}),ones(Network.maxPool)).*SwishPrime(Z{neuronIndex+ii-1});
    
    dCostdWeight{neuronIndex+ii-1} = conv2(rot90(Activations{activationIndex+ii},2),deltaL,'valid');
    dCostdBias{neuronIndex+ii-1} = deltaL;
end

for ii=(Network.numLayers-1):-1:1
    index = Network.numFilters(ii+1)*ii+1;
    neuronIndex = Network.numFilters(ii)*(ii-1)+1;
    
    for jj=1:Network.numFilters(ii)
        if ii==1
            activationIndex = 1;
        else
            activationIndex = Network.numFilters(ii)*(ii-2)+1+jj;
        end
        
        deltaL = dCostdBias{index+jj-1};
        W = Network.Weights{index+jj-1};
        
        deltaL = kron(conv2(rot90(W,2),deltaL,'full'),ones(Network.maxPool)).*SwishPrime(Z{neuronIndex+jj-1});
        dCostdWeight{neuronIndex+jj-1} = conv2(rot90(Activations{activationIndex},2),deltaL,'valid');
        dCostdBias{neuronIndex+jj-1} = deltaL;
    end
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
