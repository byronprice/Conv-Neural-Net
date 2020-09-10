% SVD_MVN.m

load('TrainingData.mat');

Images = Images';

[U,S,V] = svd(Images,'econ');

transformMat = V;

uniqueLabels = unique(Labels);
nLabels = length(uniqueLabels);

normDist = cell(nLabels,2);

for nn=1:nLabels
   currentIms = Images(Labels==uniqueLabels(nn),:);
   US = currentIms*transformMat;
   
   normDist{nn,1} = mean(US,1);
   normDist{nn,2} = cov(US);
end

N = size(Images,1);
pn = zeros(nLabels,1);

for ll=1:nLabels
   pn(ll) = sum(Labels==uniqueLabels(ll))/N;
end

nDims = 1:25;
AIC = zeros(length(nDims),1);
loglike = zeros(length(nDims),1);
trainingAccuracy = zeros(length(nDims),1);

for ii=1:length(nDims)
    classification = zeros(N,1);
    loglikelihood = 0;
    for nn=1:N
        US = Images(nn,:)*transformMat(:,1:ii);
        
        allpdfs = zeros(nLabels,1);
        for ll=1:nLabels
            mu = normDist{ll,1}(1:ii);
            sigma = normDist{ll,2}(1:ii,1:ii);
        
            allpdfs(ll) = log(mvnpdf(US,mu,sigma));
        end
        [~,ind] = max(allpdfs);

        classification(nn) = ind-1;
        
        logpost = allpdfs+log(pn)-LogSum(allpdfs+log(pn),nLabels);
        
        loglikelihood = loglikelihood+logpost(Labels(nn)+1);
    end
    trainingAccuracy(ii) = sum(classification==Labels)/N;
    AIC(ii) = 2*(ii+ii*(ii+1)/2)*nLabels-2*loglikelihood;
    loglike(ii) = loglikelihood;
%     plot(ii,loglike(ii),'.');hold on;pause(0.1);
end
figure;plot(AIC);
[~,ind] = min(AIC);

nDims = nDims(ind);

% categorize testing data
load('TestData.mat');
Images = Images';
N = size(Images,1);

classification = zeros(N,1);
for nn=1:N
    US = Images(nn,:)*transformMat(:,1:nDims);
    
    allpdfs = zeros(nLabels,1);
    for ll=1:nLabels
        mu = normDist{ll,1}(1:nDims);
        sigma = normDist{ll,2}(1:nDims,1:nDims);
        
        allpdfs(ll) = log(mvnpdf(US,mu,sigma));
    end
    [~,ind] = max(allpdfs);
    
    classification(nn) = ind-1;
end

testAccuracy = sum(classification==Labels)/N;

fprintf('Test Data Accuracy: %3.2f\n',testAccuracy);

save('MNIST_SVD.mat','AIC','normDist','nDims','testAccuracy','transformMat');