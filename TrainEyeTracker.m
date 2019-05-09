function [myNet] = TrainEyeTracker(startNetFile)
% TrainEyeTracker.m
%   Convolutional Neural Network to track pupil position and diameter of a
%   mouse

% % CREATE THE NETWORK WITH RANDOMIZED WEIGHTS AND BIASES
% maxLuminance = 255;
% imSize = [61,81];
% outSize = 4;
% filterSize = 5;
% numFilters = 10;
% numLayers = 2;
% 
% % fcNet = [500,100,25,outSize];
% fcNet = [100,100,100,25,outSize];
% 
% fprintf('\nLayers: %d  Filters: %d\n',numLayers,numFilters);
% tmp = repmat([filterSize,numFilters],[numLayers,1]);
% tmp(end,2) = 1;
% NetMatrix = cell(1,1);NetMatrix{1} = {imSize,tmp,fcNet};
% myNet = ConvNetwork(NetMatrix); % from a function
% % in this directory, builds a convolutional neural net

load(startNetFile,'Network','DIM');
imSize = DIM;maxLuminance = 255;
myNet = Network;clear Network DIM;

load('EyeTrackingGroundTruth.mat','imInfo','iterCount');
numImages = iterCount;
images = cell(numImages,1);boxes = cell(numImages,1);

for ii=1:numImages
    images{ii} = imInfo{ii,1};
    boxes{ii} = imInfo{ii,2};
end

allInds = 1:numImages;
trainingInds = randperm(numImages,round(numImages*0.75));
numTraining = length(trainingInds);

tmpInds = find(~ismember(allInds,trainingInds));
numTmp = length(tmpInds);

validationInds = tmpInds(randperm(numTmp,round(numTmp*0.6)));
numValidation = length(validationInds);

testingInds = tmpInds(~ismember(tmpInds,validationInds));
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

allValidationCosts = zeros(runs/10,1);
prevNet = cell(2,1);prevNet{1} = myNet;prevNet{2} = myNet;
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
            index = find(mnrnd(1,[0.5,0.5]));
            if index==1
               newContrast = min(rand*0.6+0.4,1);
            elseif index==2
                newContrast = 1;
            end
        end
        currentIm = (currentIm-currentMean)*newContrast+currentMean;
        desireOut = boxes{trainingInds(index)}';
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
    if mod(ii,10)==0
        [IOU] = CheckTestData(images,boxes,validationInds,maxLuminance,myNet,numValidation,imSize);
        meanIOU = mean(IOU);
        allValidationCosts(ii/10) = meanIOU;
%         plot(ii/10,meanIOU,'.');hold on;pause(1/100);
        
        if ii/10>2
            if allValidationCosts(ii/10)<allValidationCosts(ii/10-1) && allValidationCosts(ii/10)<allValidationCosts(ii/10-2)
                myNet = prevNet{1};
                break;
            end
            prevNet{1} = prevNet{2};
            prevNet{2} = myNet;
        end
        
    end
end
% COMPARE ON TEST DATA
[IOU] = CheckTestData(images,boxes,testingInds,maxLuminance,myNet,numTesting,imSize);
figure;histogram(IOU);

quants = quantile(IOU,[0.05/2,1-0.05/2]);
meanIOU = mean(IOU);
fprintf('Mean IOU: %3.3f\n',meanIOU);
fprintf('IOU 0.95 Quantiles: [%3.3f,%3.3f]\n\n',quants(1),quants(2));
end

function [IOU] = CheckTestData(images,boxes,testingInds,maxLuminance,myNet,numTesting,imSize)
IOU = zeros(numTesting,1); % intersection over union for accuracy

for ii=1:numTesting
    [Output,~] = Feedforward(images{testingInds(ii)}./maxLuminance,myNet);
    netOut = Output{end};
    netOut(3) = netOut(3)+netOut(1);
    netOut(4) = netOut(4)+netOut(2);
    netOut([1,3]) = netOut([1,3]).*imSize(2);
    netOut([2,4]) = netOut([2,4]).*imSize(1);
    desireOut = boxes{testingInds(ii)}';
    
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

% quants = quantile(IOU,[0.05/2,1-0.05/2]);
% meanIOU = mean(IOU);
% fprintf('Mean IOU: %3.3f\n',meanIOU);
% fprintf('IOU Quantiles: [%3.3f,%3.3f]\n\n',quants(1),quants(2));

end