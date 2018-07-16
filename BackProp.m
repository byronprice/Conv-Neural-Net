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

Activations = cell(1,Network.numFilters+1);
Activations{1} = Input;
for ii=2:Network.numFilters+1
    Activations{ii} = Output{ii-1};
end

dCostdWeight = cell(1,Network.numFilters+1);
dCostdBias = cell(1,Network.numFilters+1);

%deltaL = (Z{end}-DesireOutput); % linear output neuron, mean-squared error cost
deltaL = (Output{end}-DesireOutput); % cross-entropy cost with sigmoid/swish output neurons
                                % .*SwishPrime(Output{end}); % add this back for 
                                % mean-squared error cost function, unless
                                % you want the output neuron to be linear
temp = zeros(Network.numFilters,1);
for ii=1:Network.numFilters
   temp(ii) = mean(mean(Activations{ii+1}.*deltaL));
end
dCostdWeight{end} = temp;
dCostdBias{end} = mean(deltaL(:));

origDeltaL = deltaL;
for ii=1:Network.numFilters
    W = Network.Weights{Network.numFilters+1};
    deltaL = (W(ii)*origDeltaL).*SwishPrime(Z{ii});
    
    temp = zeros(Network.networkStructure(2,1),Network.networkStructure(2,2));
    for jj=1:Network.outputSize(1)
        for kk=1:Network.outputSize(2)
            temp = temp+Activations{1}(jj:jj+Network.networkStructure(2,1)-1,kk:kk+Network.networkStructure(2,2)-1)*deltaL(jj,kk);
        end
    end
    dCostdWeight{ii} = temp./prod(Network.outputSize);
    dCostdBias{ii} = mean(deltaL(:));
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
