function [myNet] = ConvNetwork(NetMatrix)
%ConvNetwork.m
 % Define an x-filter 2-layer Convolutional Neural Network as a structure array
 %   The network looks like this ...
 %     x filters are convovled with the input image, the outputs are x
 %     filter images, those x filter images are then combined non-linearly 
 %     to yield an output image ... this is then passed through a second
 %     layer of x filters
% INPUT: FilterMatrix - a matrix, such as [50,50;10,10;10,2], with the number of
%         nodes per filter, and nodes in the input and output layers ...
%         in the preceding example, 50x50 is the size of the input image,
%         10x10 is the size of each of the filters, 10 is the number of
%         filters, 2 is the size of the output layer
% OUTPUT: Structure array with randomized weights and biases representing
%           the network.  Use standard normal random variables for initial
%           values.
% Created: 2018/07/13, 24 Cummington, Boston
%  Byron Price
% Updated: 2018/02/05
%  By: Byron Price

input = zeros(NetMatrix{1}{1});
numLayers = size(NetMatrix{1}{2},1);
maxPool = 2.*ones(numLayers,1);maxPool(end) = 1;
numFilters = zeros(numLayers,1);
outNodes = NetMatrix{1}{3};

outputSize = cell(1,1);outputSize{1} = cell(numLayers,1);
inputSize = cell(1,1);inputSize{1} = cell(numLayers,1);
filterSize = cell(1,1);filterSize{1} = cell(numLayers,1);

numCalcs = 1;
for ii=1:numLayers
    filtLen = NetMatrix{1}{2}(ii,1);
    filter = zeros(filtLen,filtLen);
    inputSize{1}{ii} = size(input);
    
    check = size(conv2(input,filter,'valid'))./maxPool(ii);
    
    origFilt = filtLen;
    plusMinus = -1;step = 1;
    count = 1;
    while filtLen>0 && ~(mod(check(1),1)==0 && mod(check(2),1)==0)
       filtLen = origFilt+plusMinus*step;
       filter = zeros(filtLen,filtLen);
       check = size(conv2(input,filter,'valid'))./maxPool(ii);
       
       plusMinus = plusMinus*-1;
       step = step+mod(count-1,2);
       count = count+1;
    end
    filter = zeros(filtLen,filtLen);
    outputSize{1}{ii} = size(conv2(input,filter,'valid'))./maxPool(ii);
    filterSize{1}{ii} = filtLen;
    input = zeros(outputSize{1}{ii});
    numFilters(ii) = NetMatrix{1}{2}(ii,2);
    numCalcs = numCalcs+numFilters(ii);
end


field1 = 'Weights';
field2 = 'Biases';
field3 = 'numCalcs';
field4 = 'numFilters';
field5 = 'networkStructure';
field6 = 'outputSize';
field7 = 'numLayers';
field8 = 'inputSize';
field9 = 'maxPool';


value1 = cell(1,1);
value2 = cell(1,1);

value1{1} = cell(1,numCalcs);
value2{1} = cell(1,numCalcs);

index = 1;
for jj=1:numLayers
    currentSize = filterSize{1}{jj};
%     outSize = outputSize{1}{jj}*maxPool;
  
    for ii=1:numFilters(jj)
        if jj<0 % >1
            value1{1}{index} = normrnd(0,1/currentSize,[currentSize,currentSize,numFilters(jj-1)]);
        else
            value1{1}{index} = normrnd(0,1/currentSize,[currentSize,currentSize]);
        end
%         value2{1}{index} = normrnd(0,1/prod(outSize),outSize);
        value2{1}{index} = normrnd(0,1);
        index = index+1;
    end
%     value1{1}{index} = normrnd(0,1/sqrt(currentSize),[currentSize,1]);
%     value2{1}{index} = normrnd(0,1);
%     index = index+1;
end
fullSize = numFilters(end)*prod(outputSize{1}{end});
value1{1}{index} = normrnd(0,1/sqrt(fullSize),[fullSize,outNodes]);
value2{1}{index} = normrnd(0,1,[outNodes,1]);

value3 = numCalcs;
value4 = numFilters;
value5 = NetMatrix;
value6 = outputSize;
value7 = numLayers;
value8 = inputSize;
value9 = maxPool;

% indices = cell(1,1);indices{1} = cell(numLayers,1);
% indices2 = cell(1,1);indices2{1} = cell(numLayers,1);
% for ii=1:numLayers
%     inds = [];
%     filter = NetMatrix{1}{2}(ii,1);
%     for kk=1:outputSize{1}{ii}(1)*maxPool
%         for jj=1:outputSize{1}{ii}(2)*maxPool
%             temp = zeros(inputSize{1}{ii});
%             temp(jj:jj+filter-1,kk:kk+filter-1) = 1;
%             inds = [inds;find(temp)];
%         end
%     end
%     indices{1}{ii} = inds;
%     indices2{1}{ii} = repmat((1:filter*filter)',[length(inds)/(filter*filter),1]);
% end
% value10 = indices;
% 
% value11 = indices2;
% 
% indices = cell(1,1);indices{1} = cell(numLayers,1);
% for ii=1:numLayers
%     inds = [];
%     filter = NetMatrix{1}{2}(ii,1);
%     for kk=1:outputSize{1}{ii}(1)*maxPool
%         for jj=1:outputSize{1}{ii}(2)*maxPool
%             temp = zeros(outputSize{1}{ii}*maxPool);temp(jj,kk) = 1;
%             inds = [inds;repmat(find(temp),[filter*filter,1])];
%         end
%     end
%     indices{1}{ii} = inds;
% end
% value12 = indices;
        
myNet = struct(field1,value1,field2,value2,field3,value3,field4,value4,...
    field5,value5,field6,value6,field7,value7,field8,value8,field9,value9);
end
