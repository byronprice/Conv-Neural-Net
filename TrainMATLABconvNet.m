% TrainMATLABconvNet.m
% 
% cifar10Data = tempdir;
% 
% url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
% 
% helperCIFAR10Data.download(url,cifar10Data);
% 
% [trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);
% 
% DIM = [61,81];
% 
% height = size(trainingImages,1);width = size(trainingImages,2);numChannels = 1;
% 
% imageSize = [height width numChannels];
% inputLayer = imageInputLayer(imageSize);
% 
% numImageCategories = 10;
% 
% % Convolutional layer parameters
% filterSize = [5 5];
% numFilters = 32;
% 
% middleLayers = [
%     
% % The first convolutional layer has a bank of 32 5x5x3 filters. A
% % symmetric padding of 2 pixels is added to ensure that image borders
% % are included in the processing. This is important to avoid
% % information at the borders being washed away too early in the
% % network.
% convolution2dLayer(filterSize,numFilters,'Padding',2)
% 
% % Note that the third dimension of the filter can be omitted because it
% % is automatically deduced based on the connectivity of the network. In
% % this case because this layer follows the image layer, the third
% % dimension must be 3 to match the number of channels in the input
% % image.
% 
% % Next add the ReLU layer:
% reluLayer()
% 
% % Follow it with a max pooling layer that has a 3x3 spatial pooling area
% % and a stride of 2 pixels. This down-samples the data dimensions from
% % 32x32 to 15x15.
% maxPooling2dLayer(2,'Stride',1)
% 
% % Repeat the 3 core layers to complete the middle of the network.
% convolution2dLayer(filterSize,numFilters,'Padding',2)
% reluLayer()
% maxPooling2dLayer(2, 'Stride',1)
% 
% convolution2dLayer(filterSize,2 * numFilters,'Padding',2)
% reluLayer()
% maxPooling2dLayer(2,'Stride',1)
% 
% ];
% 
% finalLayers = [
%     
% % Add a fully connected layer with 64 output neurons. The output size of
% % this layer will be an array with a length of 64.
% fullyConnectedLayer(100)
% 
% % Add an ReLU non-linearity.
% reluLayer
% 
% % Add the last fully connected layer. At this point, the network must
% % produce 10 signals that can be used to measure whether the input image
% % belongs to one category or another. This measurement is made using the
% % subsequent loss layers.
% fullyConnectedLayer(numImageCategories)
% 
% % Add the softmax loss layer and classification layer. The final layers use
% % the output of the fully connected layer to compute the categorical
% % probability distribution over the image classes. During the training
% % process, all the network weights are tuned to minimize the loss over this
% % categorical distribution.
% softmaxLayer
% classificationLayer
% ];
% 
% layers = [
%     inputLayer
%     middleLayers
%     finalLayers
%     ];
% 
% layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);
% 
% opts = trainingOptions('sgdm', ...
%     'Momentum', 0.9, ...
%     'InitialLearnRate', 0.001, ...
%     'LearnRateSchedule', 'piecewise', ...
%     'LearnRateDropFactor', 0.1, ...
%     'LearnRateDropPeriod', 8, ...
%     'L2Regularization', 0.004, ...
%     'MaxEpochs', 1, ...
%     'MiniBatchSize', 128, ...
%     'Verbose', true);
% 
% doTraining = true;
% 
% if doTraining    
%     % Train a network.
%     newTraining = zeros(size(trainingImages,1),size(trainingImages,2),1,size(trainingImages,4),'uint8');
%     for ii=1:size(trainingImages,4)
%         tmp = trainingImages(:,:,:,ii);
%         tmp = mean(tmp,3);
%         newTraining(:,:,1,ii) = uint8(tmp);
%     end
%     cifar10Net = trainNetwork(newTraining, trainingLabels, layers, opts);
%     
%     testIms = zeros(size(testImages,1),size(testImages,2),1,size(testImages,4),'uint8');
%     
%     for ii=1:size(testImages,4)
%         tmp = testImages(:,:,:,ii);
%         tmp = mean(tmp,3);
%         testIms(:,:,1,ii) = uint8(tmp);
%     end
%     
%     yPred = classify(cifar10Net,testIms);
%     yTrue = testLabels;
%     accuracy = sum(yPred==yTrue)/numel(yTrue)
%     
%     prevAccuracy = accuracy;check = true;
%         
%     while check
%         newTraining = zeros(size(trainingImages,1),size(trainingImages,2),1,size(trainingImages,4),'uint8');
%         for ii=1:size(trainingImages,4)
%             tmp = trainingImages(:,:,:,ii);
%             tmp = mean(tmp,3);
%             newTraining(:,:,1,ii) = uint8(tmp);
%         end
%         cifar10Net = trainNetwork(newTraining, trainingLabels, cifar10Net.Layers, opts);
%         
%         testIms = zeros(size(testImages,1),size(testImages,2),1,size(testImages,4),'uint8');
%         
%         for ii=1:size(testImages,4)
%             tmp = testImages(:,:,:,ii);
%             tmp = mean(tmp,3);
%             testIms(:,:,1,ii) = uint8(tmp);
%         end
%         
%         yPred = classify(cifar10Net,testIms);
%         yTrue = testLabels;
%         accuracy = sum(yPred==yTrue)/numel(yTrue)
%         
%         check = accuracy>=prevAccuracy;
%         prevAccuracy = accuracy;
%     end
% else
%     % Load pre-trained detector for the example.
%     load('rcnnStopSigns.mat','cifar10Net')       
% end
%  
% save('cifar10Net.mat','cifar10Net','imageSize');

load('BigEyeTrackingTable.mat');

trainingIdx = randperm(iterCount,round(iterCount*0.8));
testIdx = 1:iterCount;
testIdx = find(~ismember(testIdx,trainingIdx));
    
for epochs = 1:10
    checkpointLocation = '/home/byron/Documents/Current-Projects/Conv-Neural-Net/CheckPointNets';
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',25, ...
        'InitialLearnRate', 1e-6, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', epochs, ...
        'Verbose', true,'CheckpointPath',checkpointLocation);
    
    load('BigEyeTrackingTable.mat');
    
%     trainingIdx = randperm(iterCount,round(iterCount*0.8));
%     testIdx = 1:iterCount;
%     testIdx = find(~ismember(testIdx,trainingIdx));
    
    % net = 'inceptionresnetv2'; % net = 'googlenet';
    % tmpInds = randperm(length(trainingIdx),round(length(trainingIdx)*0.2));
    rcnn = trainRCNNObjectDetector(eyeTrackTable(trainingIdx,:),'resnet101', options,...
        'PositiveOverlapRange',[0.5,1],'NegativeOVerlapRange',[0,0.25]);
    
    IOU = zeros(length(testIdx),2);
    for ii=1:length(testIdx)
        image = imread(eyeTrackTable.imageFilename{testIdx(ii)});
        bbox = eyeTrackTable.ROI{testIdx(ii)};
        
        [outbox,score,~] = detect(rcnn,image,'MiniBatchSize',25);
        if isempty(score)
            IOU(ii,1) = 0;IOU(ii,2) = 0;
        else
            [maxScore,ind] = max(score);
            netBox = outbox(ind,:);
            
            netOut = netBox;
            netOut(3) = netOut(3)+netOut(1);
            netOut(4) = netOut(4)+netOut(2);
            
            desireOut = bbox;
            desireOut(3) = desireOut(3)+desireOut(1);
            desireOut(4) = desireOut(4)+desireOut(2);
            
            trueArea = (desireOut(3)-desireOut(1))*(desireOut(4)-desireOut(2));
            netArea = max(netOut(3)-netOut(1),netOut(1)-netOut(3))*max(netOut(4)-netOut(2),netOut(2)-netOut(4));
            
            % intersection area
            xMin = max(netOut(1),desireOut(1));
            yMin = max(netOut(2),desireOut(2));
            xMax = min(netOut(3),desireOut(3));
            yMax = min(netOut(4),desireOut(4));
            
            interArea = max(0,xMax-xMin)*max(0,yMax-yMin);
            
            IOU(ii,1) = interArea/(trueArea+netArea-interArea);
            IOU(ii,2) = maxScore;
        end
    end
    prevOverallAcc = mean(IOU(:,1))
    save(sprintf('EyeTrackerRCNN%d.mat',epochs),'prevOverallAcc','rcnn','testIdx','iterCount');
end

prevOverallAcc = 0;

wildcardFilePath = fullfile(checkpointLocation,'net_checkpoint__*.mat');
contents = dir(wildcardFilePath);

filepath = fullfile(contents(1).folder,contents(1).name);
checkpoint = load(filepath);

rcnnCheckPoint = rcnnObjectDetector();
rcnnCheckPoint.RegionProposalFcn = @rcnnObjectDetector.proposeRegions;

rcnnCheckPoint.Network = checkpoint.net;

check = true;count = 0;
while check==true
%     tmpInds = randperm(length(trainingIdx),round(length(trainingIdx)*0.2));
%     rcnn = trainRCNNObjectDetector(eyeTrackTable(trainingIdx(tmpInds),:),rcnn.Network.Layers,options,...
%     'PositiveOverlapRange',[0.5,1],'NegativeOVerlapRange',[0,0.25]);
    prevNet = rcnnCheckPoint;
    
    count = count+1;
    filepath = fullfile(contents(count).folder,contents(count).name);
    checkpoint = load(filepath);
    
    rcnnCheckPoint = rcnnObjectDetector();
    rcnnCheckPoint.RegionProposalFcn = @rcnnObjectDetector.proposeRegions;
    
    rcnnCheckPoint.Network = checkpoint.net;
    
    IOU = zeros(length(testIdx),2);
    for ii=1:length(testIdx)
        image = imread(eyeTrackTable.imageFilename{testIdx(ii)});
        bbox = eyeTrackTable.ROI{testIdx(ii)};
        
        [outbox,score,~] = detect(rcnnCheckPoint,image,'MiniBatchSize',25);
        
        if isempty(score)
            IOU(ii,1) = 0;IOU(ii,2) = 0;
        else
            [maxScore,ind] = max(score);
            netBox = outbox(ind,:);
            
            netOut = netBox;
            netOut(3) = netOut(3)+netOut(1);
            netOut(4) = netOut(4)+netOut(2);
            
            desireOut = bbox;
            desireOut(3) = desireOut(3)+desireOut(1);
            desireOut(4) = desireOut(4)+desireOut(2);
            
            trueArea = (desireOut(3)-desireOut(1))*(desireOut(4)-desireOut(2));
            netArea = max(netOut(3)-netOut(1),netOut(1)-netOut(3))*max(netOut(4)-netOut(2),netOut(2)-netOut(4));
            
            % intersection area
            xMin = max(netOut(1),desireOut(1));
            yMin = max(netOut(2),desireOut(2));
            xMax = min(netOut(3),desireOut(3));
            yMax = min(netOut(4),desireOut(4));
            
            interArea = max(0,xMax-xMin)*max(0,yMax-yMin);
            
            IOU(ii,1) = interArea/(trueArea+netArea-interArea);
            IOU(ii,2) = maxScore;
        end
    end
    
    overallAcc = mean(IOU(:,1))
    
    check = overallAcc>=prevOverallAcc;
    prevOverallAcc = overallAcc;
end
rcnn = prevNet;

save('EyeTrackingRCNN.mat','rcnn','IOU','overallAcc','testIdx','trainingIdx','iterCount');