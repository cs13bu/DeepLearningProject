
%% Load
dirdata='D:\GoogleDrive\Deep learning\DataGen\Test1\';
dir_val = 'D:\GoogleDrive\Deep learning\DataGen\Test1\';

use_degen = 0;

load([dirdata,'data_fixed.mat'])
dat = reshape(permute(image,[2,3,1]),[720,1,3,length(image)]);
lab = [room_bottomwall', room_topwall', room_leftwall',room_ceiling',light_pos,cyl1_pos,cyl2_pos,cyl3_pos];

idx = find(cyl_num==3  & degenerate<1);

input_data = dat(:,:,:,idx);
ylabels = lab(idx,:);

% Order cylinders
for ii = 1:size(ylabels,1)
   xv = [ylabels(ii,8),ylabels(ii,10),ylabels(ii,12)];
   [~,I] = sort(xv);
   
   xv2 = [ylabels(ii,8:9);ylabels(ii,10:11);ylabels(ii,12:13)];
   xv2 = xv2(I,:);
   
   ylabels(ii,8:9) = xv2(1,:);
   ylabels(ii,10:11) = xv2(2,:);
   ylabels(ii,12:13) = xv2(3,:);
end


load([dirdata,'validation_fixed2.mat'])
dat = reshape(permute(image,[2,3,1]),[720,1,3,length(image)]);
lab = [room_bottomwall', room_topwall', room_leftwall',room_ceiling',light_pos,cyl1_pos,cyl2_pos,cyl3_pos];

idx = find(cyl_num==3 & degenerate<1);

test_data = dat(:,:,:,idx);
testylabels = lab(idx,:);

% Order cylinders
for ii = 1:size(testylabels,1)
   xv = [testylabels(ii,8),testylabels(ii,10),testylabels(ii,12)];
   [~,I] = sort(xv);
   
   xv2 = [testylabels(ii,8:9);testylabels(ii,10:11);testylabels(ii,12:13)];
   xv2 = xv2(I,:);
   
   testylabels(ii,8:9) = xv2(1,:);
   testylabels(ii,10:11) = xv2(2,:);
   testylabels(ii,12:13) = xv2(3,:);
end


%% Network
load('regression_network.mat')

tlayers = [convnet.Layers(1:28);
        fullyConnectedLayer(13,'Name','fc3')
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
save('regression_net_cyl3.mat')

%%


YPredicted = predict(convnet,test_data);

bottom_wall_rms = sqrt(sum(sum((YPredicted(:,1) - testylabels(:,1)).^2))/length(testylabels));
top_wall_rms = sqrt(sum(sum((YPredicted(:,2) - testylabels(:,2)).^2))/length(testylabels));
left_wall_rms = sqrt(sum(sum((YPredicted(:,3) - testylabels(:,3)).^2))/length(testylabels));
ceiling_wall_rms = sqrt(sum(sum((YPredicted(:,4) - testylabels(:,4)).^2))/length(testylabels));

light_rms = sqrt(sum(sum((YPredicted(:,5:7) - testylabels(:,5:7)).^2))/length(testylabels(:,5:7)));

cyl1_rms = sqrt(sum(sum((YPredicted(:,8:9) - testylabels(:,8:9)).^2))/length(testylabels(:,8:9)));
cyl2_rms = sqrt(sum(sum((YPredicted(:,10:11) - testylabels(:,10:11)).^2))/length(testylabels(:,10:11)));
cyl3_rms = sqrt(sum(sum((YPredicted(:,12:13) - testylabels(:,12:13)).^2))/length(testylabels(:,12:13)));

disp(['Bottom wall position RMS: ',num2str(bottom_wall_rms)])
disp(['Top wall position RMS: ',num2str(top_wall_rms)])
disp(['Left wall position RMS: ',num2str(left_wall_rms)])
disp(['Ceiling position RMS: ',num2str(ceiling_wall_rms)])
disp(['Light position RMS: ',num2str(light_rms)])
disp(['Cylinder 1 position RMS: ',num2str(cyl1_rms)])
disp(['Cylinder 2 position RMS: ',num2str(cyl2_rms)])
disp(['Cylinder 3 position RMS: ',num2str(cyl3_rms)])