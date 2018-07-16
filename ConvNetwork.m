function [myNet] = ConvNetwork(NetworkMatrix)
%ConvNetwork.m
 % Define an x-filter 1-layer Convolutional Neural Network as a structure array
 %   The network looks like this ...
 %     x filters are convovled with the input image, the outputs are x
 %     filter images, those x filter images are then combined non-linearly 
 %     to yield an output image ... the x filter images are combined to
 %     create the output image by taking a local neighborhood of pixels 2x2
 %     from each filter image
% INPUT: FilterMatrix - a matrix, such as [50,50;10,10;10,0], with the number of
%         nodes per filter, and nodes in the input and output layers ...
%         in the preceding example, 50x50 is the size of the input image,
%         10x10 is the size of each of the filters, 10 is the number of
%         numbers (0 is a place-holder), and the output layer in this case
%         is about the same size as the input layer (the size of the valid
%         convolution of the filter and the input image)
% OUTPUT: Structure array with randomized weights and biases representing
%           the network.  Use standard normal random variables for initial
%           values.
% Created: 2018/07/13, 24 Cummington, Boston
%  Byron Price
% Updated: 2018/07/13
%  By: Byron Price

input = zeros(NetworkMatrix(1,1),NetworkMatrix(1,2));
filter = ones(NetworkMatrix(2,1),NetworkMatrix(2,2));

output = conv2(input,filter,'valid');

outputSize = size(output);

field = 'Weights';
field2 = 'Biases';
field3 = 'numCalcs';
field4 = 'numFilters';
field5 = 'networkStructure';
field6 = 'outputSize';

numFilters = NetworkMatrix(3,1);
value = cell(1,1);
value2 = cell(1,1);

value{1} = cell(1,numFilters+1);
value2{1} = cell(1,numFilters+1);

for ii=1:numFilters
    value{1}{ii} = normrnd(0,1/sqrt(NetworkMatrix(2,1)*NetworkMatrix(2,2)),[NetworkMatrix(2,1),NetworkMatrix(2,2)]);
    value2{1}{ii} = normrnd(0,1);
end

value{1}{numFilters+1} = normrnd(0,1/sqrt(numFilters),[numFilters,1]);
value2{1}{numFilters+1} = normrnd(0,1);

value3 = numFilters+1;
value4 = numFilters;
value5 = NetworkMatrix;
value6 = outputSize;
myNet = struct(field,value,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6);
end