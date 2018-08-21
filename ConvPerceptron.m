% ConvPerceptron.m
% Created: 2018/07/16, 24 Cummington, Boston
%  Byron Price
% Updated: 2018/07/16
%   By: Byron Price

%% This code will implement a convolutional neural network, which will
%% hopefully be able to detect a mouse's pupil

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
%% See www.neuralnetworksanddeeplearning.com for more information.

%% CREATE THE NETWORK WITH RANDOMIZED WEIGHTS AND BIASES
imSize = 100;filterSize = 5;numFilters = 10;numLayers = 3;
NetworkStructure = [imSize,imSize;filterSize,filterSize;numFilters,numLayers];
myNet = ConvNetwork(NetworkStructure); % from a function
     % in this directory, builds a convolutional neural network
     
input = zeros(imSize,imSize);
for ii=1:numLayers
    input = conv2(input,ones(filterSize,filterSize),'valid');
end

outSize = size(input);
 
numImages = 50000;
Input = zeros(numImages,imSize,imSize);
DesireOutput = zeros(numImages,outSize(1),outSize(2));

cutSize = floor(filterSize/2)*numLayers+1;
for kk=1:numImages
    temp = ones(imSize,imSize);
    centerX = rand*30+35;
    centerY = rand*30+35;
    circleSize = 3+rand*20;
    for ii=0:imSize-1
        for jj=0:imSize-1
            dist = sqrt((ii-centerX).^2+(jj-centerY).^2);
            if dist<circleSize
                temp(ii+1,jj+1) = 0.2;
            end
        end
    end
    Input(kk,:,:) = temp;%+normrnd(0,0.1+rand*0.4,[imSize,imSize]);
    
    edges = edge(temp,'Canny');
    DesireOutput(kk,:,:) = double(edges(cutSize:end-(cutSize-1),cutSize:end-(cutSize-1)));
end

%% STOCHASTIC GRADIENT DESCENT
batchSize = 10; % make mini batches and run the algorithm
     % on those "runs" times
runs = 5e4;
eta = 0.1; % learning rate
lambda = 10; % L2 regularization parameter

numFilters = myNet.numFilters;
numLayers = myNet.numLayers;
numCalcs = myNet.numCalcs;
dCostdWeight = cell(1,numCalcs);
dCostdBias = cell(1,numCalcs);

for ii=1:runs
    indeces = ceil(rand([batchSize,1]).*(numImages-1));
    
    filterIndex = 1;
    for kk=1:numLayers
        index = (numFilters+1)*kk;
        for jj=1:numFilters
            dCostdWeight{filterIndex} = zeros(myNet.networkStructure(2,1),myNet.networkStructure(2,2));
            dCostdBias{filterIndex} = zeros(1,1);
            filterIndex = filterIndex+1;
        end
        dCostdWeight{index} = zeros(myNet.numFilters,1);
        dCostdBias{index} = zeros(1,1);
        filterIndex = filterIndex+1;
    end
    for jj=1:batchSize
        index = indeces(jj);
        [costweight,costbias] = BackProp(squeeze(Input(index,:,:)),myNet,...
        squeeze(DesireOutput(index,:,:)));
        for kk=1:numCalcs
            dCostdWeight{kk} = dCostdWeight{kk}+costweight{kk};
            dCostdBias{kk} = dCostdBias{kk}+costbias{kk};
        end
    end
    [myNet] = GradientDescent(myNet,dCostdWeight,dCostdBias,batchSize,eta,min(runs*batchSize,numImages),lambda);
%     clear indeces;% dCostdWeight dCostdBias;
%     disp(ii/runs);
end

%% COMPARE ON TEST DATA
numImages = 1000;
Input = zeros(numImages,imSize,imSize);
DesireOutput = zeros(numImages,outSize(1),outSize(2));

for kk=1:numImages
    temp = ones(imSize,imSize);
    centerX = rand*30+35;
    centerY = rand*30+35;
    circleSize = 3+rand*20;
    for ii=0:imSize-1
        for jj=0:imSize-1
            dist = sqrt((ii-centerX).^2+(jj-centerY).^2);
            if dist<circleSize
                temp(ii+1,jj+1) = 0.2;
            end
        end
    end
    Input(kk,:,:) = temp;%+normrnd(0,0.2+rand*0.5,[imSize,imSize]); % more noise in test case
    
    edges = edge(temp,'Canny');
    DesireOutput(kk,:,:) = double(edges(cutSize:end-(cutSize-1),cutSize:end-(cutSize-1)));
end

accuracy = zeros(numImages,1);
truePositives = zeros(numImages,1);
falsePositives = zeros(numImages,1);
for ii=1:numImages
    [Output,~] = Feedforward(squeeze(Input(ii,:,:)),myNet);
    guessIm = Output{end};
    logicalIm = guessIm>0.1;
    new = squeeze(DesireOutput(ii,:,:))+logicalIm;
    vals = find(new==2 | new==0);
    accuracy(ii) = length(vals)./prod(outSize);
    
    trueVals = find(squeeze(DesireOutput(ii,:,:))==1);
    falseVals = find(squeeze(DesireOutput(ii,:,:))==0);
    
    vals = find(logicalIm==1);
    truePositives(ii) = sum(ismember(trueVals,vals))./length(trueVals);
    
    falsePositives(ii) = sum(ismember(falseVals,vals))./length(falseVals);
end
Accuracy = mean(accuracy);
TPR = mean(truePositives);
FPR = mean(falsePositives);
fprintf('Accuracy: %3.3f\n',Accuracy);
fprintf('True Positive Rate: %3.3f\n',TPR);
fprintf('False Positive Rate: %3.3f\n',FPR);
figure;histogram(accuracy);title('Accuracy');
figure;histogram(truePositives);title('True Positive Rate');
figure;histogram(falsePositives);title('False Positive Rate');
