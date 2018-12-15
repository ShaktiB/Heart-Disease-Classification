clear all 
close all

%%
Eta = 0.1; % Learning rate 
Theta = 0.001;
MaxNoOfIteration = 300;
Problem = 2;

%% Loading data 
load('heartdisease.mat');

Data = DataLab3;

[J,w] = NeuralNetwork(Eta ,Theta, MaxNoOfIteration, Problem, Data);
