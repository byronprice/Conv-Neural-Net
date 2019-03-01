function [y] = Swish(Z)
%Swish.m
%   see Searching for Activation Functions, Ramachandran et al. Google
%     Brain
% Calculate the Swish function for an input
%  fix the tunable parameter beta to be 1
y = Z./(1+exp(-Z)); % beta would be multiplied by Z in the sigmoid
end