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
dCostdWeight{end} = Activations{end}(:)*deltaL';
dCostdBias{end} = deltaL;

tmpDeltaL = zeros(Network.outputSize(end,:));
tmpWeights = zeros(Network.numFilters,1);
for jj=1:Network.numFilters
    tmp = reshape(Network.Weights{end}*deltaL,Network.outputSize(end,:)).*SwishPrime(Z{end-1});
    tmpWeights(jj) = sum(sum(Activations{end-jj}.*tmp));
    tmpDeltaL = tmpDeltaL+tmp;
end

dCostdWeight{end-1} = tmpWeights;
dCostdBias{end-1} = sum(tmpDeltaL(:));

deltaL = tmpDeltaL;

for ll=Network.numLayers:-1:1
    activationIndex = (Network.numFilters+2)*(ll-1)+1;
    index = (Network.numFilters+2)*ll;
    
    if ll==Network.numLayers
        
    else
        tmpDeltaL = zeros(Network.outputSize(ll,:));
        tmpWeights = zeros(Network.numFilters,1);
        for jj=1:Network.numFilters
            W = Network.Weights{index+jj};
            tmp = (W*deltaL(:,:,jj)).*SwishPrime(Z{index});
            tmpWeights(jj) = sum(sum(Activations{index-jj}.*tmp));
            tmpDeltaL = tmpDeltaL+tmp;
        end
        
        dCostdWeight{index} = tmpWeights;
        dCostdBias{index} = sum(tmpDeltaL(:));
        
        deltaL = tmpDeltaL;
    end

    origDeltaL = deltaL;
    W = Network.Weights{index};
    deltaL = zeros([size(Z{activationIndex+ii-1}),Network.numFilters]);
    for ii=1:Network.numFilters
        tmp = W(ii)*origDeltaL;
        tmpDeltaL = kron(tmp,ones(Network.maxPool)).*SwishPrime(Z{activationIndex+ii-1});

        dCostdWeight{activationIndex+ii-1} = rot90(reshape(accumarray(Network.idx2,Activations{activationIndex}(Network.idx1).*tmpDeltaL(Network.idx3)),...
            [Network.networkStructure(2,1),Network.networkStructure(2,2)]),2);
        dCostdBias{activationIndex+ii-1} = sum(tmpDeltaL(:));
        
        deltaL(:,:,ii) = tmpDeltaL;
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
