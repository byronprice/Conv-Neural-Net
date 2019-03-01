% TrainImageNet.m

for numLayers = 2:3
    for numFilters = 2:10
% CREATE THE NETWORK WITH RANDOMIZED WEIGHTS AND BIASES
outSize = 4;
filterSize = 5;
% numFilters = 10;

fprintf('Layers: %d  Filter Number: %d\n',numLayers,numFilters);
NetMatrix = cell(1,1);NetMatrix{1} = {[61,81],repmat([filterSize,numFilters],[numLayers,1]),outSize};
myNet = ConvNetwork(NetMatrix); % from a function
% in this directory, builds a convolutional neural net

load('ImageNetData.mat');
numImages = length(images);

allInds = 1:numImages;
trainingInds = randperm(numImages,round(numImages*0.8));
numTraining = length(trainingInds);
testingInds = find(~ismember(allInds,trainingInds));
numTesting = length(testingInds);

% STOCHASTIC GRADIENT DESCENT
batchSize = 10; % make mini batches and run the algorithm
% on those "runs" times
runs = 5e3;
eta = 0.005; % learning rate
lambda = 10; % L2 regularization parameter

numCalcs = myNet.numCalcs;
dCostdWeight = cell(1,numCalcs);
dCostdBias = cell(1,numCalcs);

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
        if ii<numTraining
            newContrast = 1;
        else
            newContrast = rand*0.5+0.5;
        end
        currentIm = (currentIm-currentMean)*newContrast+currentMean;
        desireOut = boxes{trainingInds(index),4}';
        desireOut(3) = desireOut(3)-desireOut(1);
        desireOut(4) = desireOut(4)-desireOut(2);
        [costweight,costbias] = BackProp(currentIm,myNet,desireOut);
        for kk=1:numCalcs
            dCostdWeight{kk} = dCostdWeight{kk}+costweight{kk};
            dCostdBias{kk} = dCostdBias{kk}+costbias{kk};
        end
    end
    [myNet] = GradientDescent(myNet,dCostdWeight,dCostdBias,batchSize,eta,numTraining,lambda);
    %     clear indeces;% dCostdWeight dCostdBias;
end

% COMPARE ON TEST DATA

IOU = zeros(numTesting,1); % intersection over union for accuracy
count = 0;
for ii=1:numTesting
    [Output,Z] = Feedforward(images{testingInds(ii)},myNet);
    netOut = Output{end};
    netOut(3) = netOut(3)+netOut(1);
    netOut(4) = netOut(4)+netOut(2);
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

fprintf('Mean IOU: %3.3f\n\n',mean(IOU));

    end
end
% for ii=1:5
%     index = ceil(rand*(numImages-1));
%     digit = classifiedVals(index);
%     image = reshape(Images(:,index),[28,28]);
%     figure();imagesc(image);title(sprintf('Classified as a(n) %i',digit));
%     colormap gray;
% end
