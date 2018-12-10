
%% Load
dirdata='D:\GoogleDrive\Deep learning\DataGen\Test1\';
dir_val = 'D:\GoogleDrive\Deep learning\DataGen\Test1\';

use_degen = 0;

load([dirdata,'data_fixed.mat'])
dat = reshape(permute(image,[2,3,1]),[720,1,3,length(image)]);
lab = [room_bottomwall', room_topwall', room_leftwall',room_ceiling',light_pos];

idx = find(cyl_num==0  & degenerate<1);

input_data = dat(:,:,:,idx);
ylabels = lab(idx,:);

load([dirdata,'validation_fixed2.mat'])
dat = reshape(permute(image,[2,3,1]),[720,1,3,length(image)]);
lab = [room_bottomwall', room_topwall', room_leftwall',room_ceiling',light_pos];

idx = find(cyl_num==0 & degenerate<1);

test_data = dat(:,:,:,idx);
testylabels = lab(idx,:);

%% Network
load('regression_network.mat')

tlayers = [convnet.Layers(1:28);
        fullyConnectedLayer(7,'Name','fc3')
    regressionLayer];


%% Train
options = trainingOptions('adam', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',0.5e-3, ...
    'ValidationPatience', 60,...
    'LearnRateSchedule','piecewise', ...%'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{test_data,testylabels}, ...
    'ValidationFrequency',200, ...
    'Plots','training-progress', ...
    'L2Regularization',0.0001,...
    'Verbose',false);

[convnet,info] = trainNetwork(input_data,ylabels,tlayers,options);

%%
save('regression_net_cyl0.mat')

%%

YPredicted = predict(convnet,test_data);

bottom_wall_rms = sqrt(sum(sum((YPredicted(:,1) - testylabels(:,1)).^2))/length(testylabels));
top_wall_rms = sqrt(sum(sum((YPredicted(:,2) - testylabels(:,2)).^2))/length(testylabels));
left_wall_rms = sqrt(sum(sum((YPredicted(:,3) - testylabels(:,3)).^2))/length(testylabels));
ceiling_wall_rms = sqrt(sum(sum((YPredicted(:,4) - testylabels(:,4)).^2))/length(testylabels));

light_rms = sqrt(sum(sum((YPredicted(:,5:7) - testylabels(:,5:7)).^2))/length(testylabels(:,5:7)));

disp(['Bottom wall position RMS: ',num2str(bottom_wall_rms)])
disp(['Top wall position RMS: ',num2str(top_wall_rms)])
disp(['Left wall position RMS: ',num2str(left_wall_rms)])
disp(['Ceiling position RMS: ',num2str(ceiling_wall_rms)])
disp(['Light position RMS: ',num2str(light_rms)])