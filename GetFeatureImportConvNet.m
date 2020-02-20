function [featImport,outputDim] = GetFeatureImportConvNet(Network,Input,outputDim)
% GetFeatureImport.m
%   feature importance, aka axiomatic feature attribution, from paper
%  Axiomatic Attribution for Deep Networks, Sundararajan et al. 2017
%    works for feedfoward convolutional network described in this repository
if nargin<3
    [Output,~] = Feedforward(Input,Network);
    [~,outputDim] = max(Output{end});
end

baselineValue = 0.5; % 0.5 for greyscale baseline, 0 for black

numGradCalcs = 1e3;
% alpha = (1:numGradCalcs)./numGradCalcs;

alpha = linspace(0,1,numGradCalcs);

baselineIm = baselineValue.*ones(size(Input));
integral = 0;
for jj=1:numGradCalcs
    [gradient] = FIBackProp(baselineIm+alpha(jj).*(Input-baselineIm),Network,outputDim);
    integral = integral+gradient./numGradCalcs;
end
featImport = (Input-baselineIm).*integral;

%  these two should be approximately equal ... if they are not, then make
%    numGradCalcs bigger
[Output2,~] = Feedforward(baselineIm,Network);
FxDiff = Output{end}(outputDim)-Output2{end}(outputDim);
integratedGrads = sum(featImport);

tolerance = 1e-2;
if (abs(FxDiff-integratedGrads)/abs(FxDiff))>tolerance
    disp('Increase number of gradient calculations');
end

end

function [gradient] = FIBackProp(Input,Network,outputDim)
%FIBackProp.m
%  See http://neuralnetworksanddeeplearning.com/chap2.html for more
%   information on the back propagation algorithm.
%INPUT: Input - input vector to the neural network
%       Network - structure array representing the network
%          see the functions Network and Feedforward
%
%OUTPUT: gradient - gradient of the output with respect to the input
%
% Created: 2016/02/05, 24 Cummington, Boston
%  Byron Price
% Updated: 2016/10/13
%  By: Byron Price

[Output,Z] = Feedforward(Input,Network);

% Activations = cell(1,Network.numCalcs);
% Activations{1} = Input;
% for ii=2:Network.numCalcs
%     Activations{ii} = Output{ii-1};
% end

for ii=Network.numCalcs:-1:Network.numCalcs-Network.numFC+1
    if ii==Network.numCalcs
        gradient = Network.Weights{ii}(:,outputDim).*SwishPrime(Z{ii}(outputDim));
    else
        gradient = Network.Weights{ii}*(gradient.*SwishPrime(Z{ii}));
    end
end

end