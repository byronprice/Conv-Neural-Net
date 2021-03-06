function [y] = SwishPrime(Z)
%SwishPrime.m
% Calculate the derivative of the Swish function for an input
%  fix the tunable parameter beta to be 1
sigmoid = 1./(1+exp(-Z));
y = Z.*sigmoid+sigmoid.*(1-Z.*sigmoid);
end