% ConvPerceptron.m
% Created: 2018/08/21, 24 Cummington, Boston
%  Byron Price
% Updated: 2018/08/21
%   By: Byron Price

%% This code will implement a convolutional neural network that 
%%  will be capable of recognizing handwritten text

%% Simple perceptron rule
%%   If w is a connection strength vector (or weight) 
%%   and x is an input vector, then 
%%     output = 0 if w*x + b <= 0
%%     output = 1 if w*x + b > 0 
%%   where * is the dot product of all inputs by their
%%   respective weights

%% Sigmoid perceptron
%%  If x is an input vector and w a connection strength
%%  vector, then output = 1/(1+exp(-(w*x+b)))
%%
%%  Here, I'm using an activation function known as the Swish, see the function
%%   Swish.m for more information.
%% See www.neuralnetworksanddeeplearning.com for more information.


load('TrainingData.mat')

numImages = size(Images,2);
numPixels = sqrt(size(Images,1));

expandSet = numImages+20000;
newIms = zeros(size(Images,1),expandSet);
newIms(:,1:numImages) = Images;

newLabels = zeros(expandSet,1);
newLabels(1:numImages) = Labels;

% expand training data with relatively small rotations of the original
%  images (5 to 20 degrees in either direction)
for ii=1:20000
    index = randperm(numImages,1);
    tmp = reshape(Images(:,index),[numPixels,numPixels]);
    rotVal = (5+rand*15)*(binornd(1,0.5)*2-1);
    new = imrotate(tmp,rotVal,'bilinear','crop');
    newIms(:,numImages+ii) = new(:);
    newLabels(numImages+ii) = Labels(index);
end
Images = newIms;
Labels = newLabels;

numImages = length(Labels);

% add a bit of noise to some of the images
for ii=1:8000
    index = randperm(numImages,1);
    Images(:,index) = Images(:,index)+(rand([numPixels*numPixels,1])*0.2-0.1);
    Images(:,index) = min(max(Images(:,index),0),1);
end

% CREATE THE NETWORK WITH RANDOMIZED WEIGHTS AND BIASES
numDigits = 10;
filterSize = 5;
numFilters = 10;
numLayers = 2;

imSize = [numPixels,numPixels];
tmp = repmat([filterSize,numFilters],[numLayers,1]);
tmp(end,2) = 5;
fcNet = numDigits;

NetMatrix = cell(1,1);NetMatrix{1} = {imSize,tmp,fcNet};
myNet = ConvNetwork(NetMatrix); % from a function
% in this directory, builds a convolutional neural net

DesireOutput = zeros(numDigits,numImages);

for ii=1:numImages
    numVector = zeros(numDigits,1);
    for jj=1:numDigits
        if Labels(ii) == jj-1
            numVector(jj) = 1;
            DesireOutput(:,ii) = numVector;
        end
    end
end

% STOCHASTIC GRADIENT DESCENT
batchSize = 10; % make mini batches and run the algorithm
% on those "runs" times
runs = 1e5;
eta = 0.001; % learning rate
lambda = 1; % L2 regularization parameter

numCalcs = myNet.numCalcs;
dCostdWeight = cell(1,numCalcs);
dCostdBias = cell(1,numCalcs);

for ii=1:runs
    indices = ceil(rand([batchSize,1]).*(numImages-1));
    for jj=1:numCalcs
        dCostdWeight{jj} = zeros(size(myNet.Weights{jj}));
        dCostdBias{jj} = zeros(size(myNet.Biases{jj}));
    end
    for jj=1:batchSize
        index = indices(jj);
        if binornd(1,1e-3)
            [costweight,costbias] = BackProp(0.5.*ones([numPixels,numPixels]),myNet,...
                zeros(numDigits,1));
        else
            [costweight,costbias] = BackProp(reshape(Images(:,index),[numPixels,numPixels]),myNet,...
                DesireOutput(:,index));
        end
        for kk=1:numCalcs
            dCostdWeight{kk} = dCostdWeight{kk}+costweight{kk};
            dCostdBias{kk} = dCostdBias{kk}+costbias{kk};
        end
    end
    [myNet] = GradientDescent(myNet,dCostdWeight,dCostdBias,batchSize,eta,numImages,lambda);
    %     clear indeces;% dCostdWeight dCostdBias;
end

% COMPARE ON TEST DATA
clear Images Labels;
load('TestData.mat')
numImages = size(Images,2);
numPixels = sqrt(size(Images,1));
numDigits = 10;

DesireOutput = zeros(numDigits,numImages);

for ii=1:numImages
    numVector = zeros(numDigits,1);
    for jj=1:numDigits
        if Labels(ii) == jj-1
            numVector(jj) = 1;
            DesireOutput(:,ii) = numVector;
        end
    end
end

classifiedVals = zeros(numImages,1);
count = 0;
for ii=1:numImages
    [Output,Z] = Feedforward(reshape(Images(:,ii),[numPixels,numPixels]),myNet);
    [~,realVal] = max(DesireOutput(:,ii));
    [~,netVal] = max(Output{end});
    classifiedVals(ii) = netVal-1;
    if realVal == netVal
        count = count+1;
    end
end
Accuracy = count/numImages;

fprintf('Accuracy: %3.3f\n\n',Accuracy);

% for ii=1:5
%     index = ceil(rand*(numImages-1));
%     digit = classifiedVals(index);
%     image = reshape(Images(:,index),[28,28]);
%     figure();imagesc(image);title(sprintf('Classified as a(n) %i',digit));
%     colormap gray;
% end
