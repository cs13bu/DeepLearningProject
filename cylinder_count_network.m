%% Load
dirdata ='D:\GoogleDrive\Deep learning\DataGen\Test1\';
dir_val = 'D:\GoogleDrive\Deep learning\DataGen\Test1\';

use_degen = 0;

load([dirdata,'data_fixed.mat'])
input_data = reshape(permute(image,[2,3,1]),[720,1,3,length(image)]);
ylabels = cyl_num';

if use_degen ==0
    idx = find(degenerate>0);
    input_data(:,:,:,idx) = [];
    ylabels(idx) = [];
end

ylabels =  categorical(ylabels);

clear degenerate
load([dirdata,'validation_fixed2.mat'])
test_data = reshape(permute(image,[2,3,1]),[720,1,3,length(image)]);
testylabels = cyl_num';

if use_degen == 0
    idx = find(degenerate>0);
    test_data(:,:,:,idx) = [];
    testylabels(idx) = [];
end

testylabels =  categorical(testylabels);

classWeights = 1./countcats(ylabels);
classWeights = classWeights'/mean(classWeights);


%% Network
layers = [
    imageInputLayer([720,1,3],'Name','Input')
    
    convolution2dLayer([11,1],30,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer([2,1],'Stride',2,'Name','avpool1')
    
    convolution2dLayer([5,1],20,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer([2,1],'Stride',2,'Name','avpool2')
    
    convolution2dLayer([3,1],12,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    
    convolution2dLayer([3,1],12,'Padding','same','Name','conv_4')
    batchNormalizationLayer('Name','BN_4')
    reluLayer('Name','relu_4')
    
%     fullyConnectedLayer(600,'Name','fc1')
%     reluLayer('Name','Relu3')
%     batchNormalizationLayer('Name','BN_2')
%     
%     fullyConnectedLayer(300,'Name','fc1')
%     reluLayer('Name','Relu3')
%     batchNormalizationLayer('Name','BN_2')
    
    %
    fullyConnectedLayer(200,'Name','fc1')
    reluLayer('Name','Relu5')
    batchNormalizationLayer('Name','BN_5')
   
    dropoutLayer(0.7,'Name','Drop1')
    
    fullyConnectedLayer(100,'Name','fc2')
    reluLayer('Name','Relu6')
    batchNormalizationLayer('Name','BN_6')
    
   dropoutLayer(0.7,'Name','Drop2')
    
    fullyConnectedLayer(80,'Name','fc3')
    reluLayer('Name','Relu7')
    batchNormalizationLayer('Name','BN_7')
    
    
%     fullyConnectedLayer(35,'Name','fc2')
%     reluLayer('Name','Relu4')
%     batchNormalizationLayer('Name','BN_3')
    
    fullyConnectedLayer(40,'Name','fc4')
    reluLayer('Name','Relu8')
    batchNormalizationLayer('Name','BN_8')
    
%     fullyConnectedLayer(20,'Name','fc2')
%     reluLayer('Name','Relu4')
%     batchNormalizationLayer('Name','BN_3')
    
    fullyConnectedLayer(4,'Name','fc5')
    softmaxLayer('Name','soft')
    weightedClassificationLayer(classWeights)];

%% Train
options = trainingOptions('adam', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',0.05e-3, ...
    'ValidationPatience', 60,...
    'LearnRateSchedule','piecewise', ...%'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{test_data,testylabels}, ...
    'ValidationFrequency',200, ...
    'Plots','training-progress', ...
    'L2Regularization',0.0001,...
    'Verbose',false);

[convnet,info] = trainNetwork(input_data,ylabels,layers,options);

%%
YPredicted = classify(convnet,test_data);

test = zeros(length(YPredicted),4);
for i = 1:length(YPredicted)
    test(i,testylabels(i)) = 1;
end

pred = zeros(length(YPredicted),4);
for i = 1:length(YPredicted)
    pred(i,YPredicted(i)) = 1;
end

plotconfusion(test',pred')

%%
save('regression_count_weighted_final.mat')
