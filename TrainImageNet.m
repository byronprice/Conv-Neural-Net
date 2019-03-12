function [myNet] = TrainImageNet()
% TrainImageNet.m
%   Convolutional Neural Network

% CREATE THE NETWORK WITH RANDOMIZED WEIGHTS AND BIASES
maxLuminance = 255;
imSize = [61,81];
outSize = 4;
filterSize = 5;
numFilters = 15;
numLayers = 3;

% fcNet = [500,100,25,outSize];
fcNet = [500,50,outSize];

fprintf('\nLayers: %d  Filters: %d\n',numLayers,numFilters);
tmp = repmat([filterSize,numFilters],[numLayers,1]);
tmp(end,2) = 1;
NetMatrix = cell(1,1);NetMatrix{1} = {imSize,tmp,fcNet};
myNet = ConvNetwork(NetMatrix); % from a function
% in this directory, builds a convolutional neural net

% load('EyeTrackerConvNet.mat','Network');
% myNet = Network;clear Network;

load('ImageNetData.mat','images','boxes');
numImages = length(images);

allInds = 1:numImages;
trainingInds = randperm(numImages,round(numImages*0.8));
numTraining = length(trainingInds);
testingInds = find(~ismember(allInds,trainingInds));
numTesting = length(testingInds);

% STOCHASTIC GRADIENT DESCENT
batchSize = 10; % make mini batches and run the algorithm
% on those "runs" times
runs = 1e5;
eta = 1e-5; % learning rate
lambda = 10; % L2 regularization parameter

numCalcs = myNet.numCalcs;
dCostdWeight = cell(1,numCalcs);
dCostdBias = cell(1,numCalcs);

allTestCosts = zeros(runs/1e2,1);
for ii=1:runs
    indices = ceil(rand([batchSize,1]).*(numTraining-1));
    for jj=1:numCalcs
        dCostdWeight{jj} = zeros(size(myNet.Weights{jj}));
        dCostdBias{jj} = zeros(size(myNet.Biases{jj}));
    end
    for jj=1:batchSize
        index = indices(jj);
        currentIm = images{trainingInds(index)};
        currentMean = mean(currentIm(:));
        if ii<2*numTraining
            newContrast = 1;
        else
            newContrast = min(rand*0.7+0.4,1);
            
            if binornd(1,0.25)
               currentIm = maxLuminance-currentIm; 
               currentMean = mean(currentIm(:));
            end
        end
        currentIm = (currentIm-currentMean)*newContrast+currentMean;
        desireOut = boxes{trainingInds(index),4}';
        desireOut([1,3]) = desireOut([1,3])./imSize(2);
        desireOut([2,4]) = desireOut([2,4])./imSize(1);
        desireOut(3) = desireOut(3)-desireOut(1);
        desireOut(4) = desireOut(4)-desireOut(2);
        [costweight,costbias] = BackProp(currentIm./maxLuminance,myNet,desireOut);
        for kk=1:numCalcs
            dCostdWeight{kk} = dCostdWeight{kk}+costweight{kk};
            dCostdBias{kk} = dCostdBias{kk}+costbias{kk};
        end
    end
    [myNet] = GradientDescent(myNet,dCostdWeight,dCostdBias,batchSize,eta,numTraining,lambda);
   if mod(ii,1e2)==0
        [meanIOU] = CheckTestData(images,boxes,testingInds,maxLuminance,myNet,numTesting,imSize);
        allTestCosts(ii/1e2) = meanIOU;
        plot(ii/5e2,meanIOU,'.');hold on;pause(1/100);
    end
end
% COMPARE ON TEST DATA
end

function [meanIOU] = CheckTestData(images,boxes,testingInds,maxLuminance,myNet,numTesting,imSize)
IOU = zeros(numTesting,1); % intersection over union for accuracy

for ii=1:numTesting
    [Output,~] = Feedforward(images{testingInds(ii)}./maxLuminance,myNet);
    netOut = Output{end};
    netOut(3) = netOut(3)+netOut(1);
    netOut(4) = netOut(4)+netOut(2);
    netOut([1,3]) = netOut([1,3]).*imSize(2);
    netOut([2,4]) = netOut([2,4]).*imSize(1);
    desireOut = boxes{testingInds(ii),4}';
    
    trueArea = (desireOut(3)-desireOut(1))*(desireOut(4)-desireOut(2));
    netArea = max(netOut(3)-netOut(1),netOut(1)-netOut(3))*max(netOut(4)-netOut(2),netOut(2)-netOut(4));
    
    % intersection area
    xMin = max(netOut(1),desireOut(1));
    yMin = max(netOut(2),desireOut(2));
    xMax = min(netOut(3),desireOut(3));
    yMax = min(netOut(4),desireOut(4));
    
    interArea = max(0,xMax-xMin)*max(0,yMax-yMin);
    
    IOU(ii) = interArea/(trueArea+netArea-interArea);
end

quants = quantile(IOU,[0.05/2,1-0.05/2]);
meanIOU = mean(IOU);
% fprintf('Mean IOU: %3.3f\n',meanIOU);
% fprintf('IOU Quantiles: [%3.3f,%3.3f]\n\n',quants(1),quants(2));

end