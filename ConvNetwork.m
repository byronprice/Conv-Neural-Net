function [myNet] = ConvNetwork(NetworkMatrix)
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
% Updated: 2018/08/30
%  By: Byron Price

input = zeros(NetworkMatrix(1,1),NetworkMatrix(1,2));
filter = ones(NetworkMatrix(2,1),NetworkMatrix(2,2));
numLayers = 1;maxPool = 2;
numFilters = NetworkMatrix(3,1);
outNodes = NetworkMatrix(3,2);

outputSize = size(conv2(input,filter,'valid'))./maxPool;
inputSize = size(input);

field = 'Weights';
field2 = 'Biases';
field3 = 'numCalcs';
field4 = 'numFilters';
field5 = 'networkStructure';
field6 = 'outputSize';
field7 = 'numLayers';
field8 = 'inputSize';
field9 = 'maxPool';
field10 = 'idx1';
field11 = 'idx2';
field12 = 'idx3';


value = cell(1,1);
value2 = cell(1,1);

value{1} = cell(1,(numFilters+1)*numLayers);
value2{1} = cell(1,(numFilters+1)*numLayers);

index = 1;
for jj=1:numLayers
    for ii=1:numFilters
        value{1}{index} = normrnd(0,1/sqrt(NetworkMatrix(2,1)*NetworkMatrix(2,2)),[NetworkMatrix(2,1),NetworkMatrix(2,2)]);
        value2{1}{index} = normrnd(0,1);
        index = index+1;
    end
    fullSize = numFilters*prod(outputSize);
    value{1}{index} = normrnd(0,1/sqrt(fullSize),[fullSize,outNodes]);
    value2{1}{index} = normrnd(0,1,[outNodes,1]);
    index = index+1;
end

value3 = (numFilters+1)*numLayers;
value4 = numFilters;
value5 = NetworkMatrix;
value6 = outputSize;
value7 = numLayers;
value8 = inputSize;
value9 = maxPool;

inds = [];
for jj=1:outputSize(1)*maxPool
    for kk=1:outputSize(2)*maxPool
        temp = zeros(inputSize);
        temp(jj:jj+NetworkMatrix(2,1)-1,kk:kk+NetworkMatrix(2,2)-1) = 1;
        inds = [inds;find(temp)];
    end
end
value10 = inds;

inds = repmat((1:numel(filter))',[length(inds)/numel(filter),1]);
value11 = inds;

inds = [];
for jj=1:outputSize(1)*maxPool
    for kk=1:outputSize(2)*maxPool
        temp = zeros(outputSize*maxPool);temp(jj,kk) = 1;
        inds = [inds;repmat(find(temp),[numel(filter),1])];
    end
end
value12 = inds;
        
myNet = struct(field,value,field2,value2,field3,value3,field4,value4,...
    field5,value5,field6,value6,field7,value7,field8,value8,field9,value9,...
    field10,value10,field11,value11,field12,value12);
end
