clear all;
clc;
close all;

% Read data
[dataTable, data, x, y, xTrain, yTrain, xTest, yTest] = getData("Student_Performance.csv");

% Create Neural Network Structure
inputSize = size(xTrain, 2);
structure = [inputSize, 10, 5, 1];
layerCount = size(structure, 2);

% Initialize weights and biases
w = {};
b = {};

for iL = 2:layerCount
    w{iL-1} = rand([structure(iL), structure(iL-1)]) * 2 - 1;
    b{iL-1} = rand([structure(iL), 1]) * 2 - 1;
end


% Hyperparameters
learningRate = 0.0001;
epochs = 10000;

% Activation function and its derivative
sigmoid = @(z) 1 ./ (1 + exp(-z));
sigmoidDerivative = @(a) a .* (1 - a);

epochLossGraph = [];

tic();
% Train the Neural Network
for epoch = 1:epochs
    % Forward Propagation
    z = {xTrain'};
    a = {xTrain'};
    for iL = 2:layerCount
        z{iL} = w{iL-1} * a{iL-1} + b{iL-1};
        a{iL} = sigmoid(z{iL});
    end
    
    % Compute loss
    output = a{end};
    loss = -mean(yTrain' .* log(output) + (1 - yTrain') .* log(1 - output));
    epochLossGraph = [epochLossGraph, loss];
    
    % Backward Propagation
    delta = {};
    gradW = {};
    gradB = {};
    
    % Output layer error
    delta{layerCount} = (output - yTrain') .* sigmoidDerivative(output);
    
    % Compute gradients and backpropagation
    for iL = layerCount-1:-1:1
        gradW{iL} = delta{iL+1} * a{iL}';
        gradB{iL} = sum(delta{iL+1}, 2);
        
        if iL > 1
            delta{iL} = (w{iL}' * delta{iL+1}) .* sigmoidDerivative(a{iL});
        end
    end
    
    % Update weights and biases
    for iL = 1:layerCount-1
        w{iL} = w{iL} - learningRate * gradW{iL};
        b{iL} = b{iL} - learningRate * gradB{iL};
    end
    
    % Display loss every 100 epochs
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.6f\n', epoch, loss);
    end
end
trainTime = toc();

% Test the network on the test set
zTest = {xTest'};
aTest = {xTest'};
for iL = 2:layerCount
    zTest{iL} = w{iL-1} * aTest{iL-1} + b{iL-1};
    aTest{iL} = sigmoid(zTest{iL});
end

% Compute test loss
outputTest = aTest{end};
testLoss = -mean(yTrain' .* log(output) + (1 - yTrain') .* log(1 - output));
fprintf('Test Loss: %.6f\n', testLoss);

% Display time, accuracy and epoch-loss graph
r1 = outputTest' > 0.50;
r2 = yTest > 0.50;
predTrue = size(r1,1) - sum(xor(r1,r2));
acc = predTrue/size(r1,1);
fprintf("Train Time (s): %.6f\n", trainTime);
fprintf("Accuracy: %.6f\n", acc);
plot([1:epochs], epochLossGraph, 'ko');
title("Epoch-Loss Graph");