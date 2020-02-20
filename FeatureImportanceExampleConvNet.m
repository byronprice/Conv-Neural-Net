% FeatureImportanceExampleConvNet.m

load('TestData.mat');
numImages = size(Images,2);
numPixels = sqrt(size(Images,1));
DIM = [numPixels,numPixels];
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

load('MNIST_ConvNet.mat','Network');

whichIms = randperm(numImages,10);
for ii=1:10
    tmpIm = reshape(Images(:,whichIms(ii)),DIM);
    [featImport,outputDim] = GetFeatureImportConvNet(Network,tmpIm);
    
    figure;subplot(1,2,1);
    imagesc(reshape(Images(:,whichIms(ii)),[28,28]));
    title(sprintf('Ground Truth: %d',Labels(whichIms(ii))));
    colormap('gray');
    subplot(1,2,2);
    imagesc(reshape(featImport,[28,28]));
    tmp = max(featImport);
    caxis([-tmp tmp]);title(sprintf('Model Output: %d',outputDim-1));
    colormap('gray');
end