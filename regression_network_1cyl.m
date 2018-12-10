%% Load
dirdata='D:\GoogleDrive\Deep learning\DataGen\Test1\';
dir_val = 'D:\GoogleDrive\Deep learning\DataGen\Test1\';

use_degen = 0;

load([dirdata,'data_fixed.mat'])
dat = reshape(permute(image,[2,3,1]),[720,1,3,length(image)]);
lab = [room_bottomwall', room_topwall', room_leftwall',room_ceiling',light_pos,cyl1_pos];

idx = find(cyl_num==1  & degenerate<1);

input_data = dat(:,:,:,idx);
ylabels = lab(idx,:);


load([dirdata,'validation_fixed2.mat'])
dat = reshape(permute(image,[2,3,1]),[720,1,3,length(image)]);
lab = [room_bottomwall', room_topwall', room_leftwall',room_ceiling',light_pos,cyl1_pos];

idx = find(cyl_num==1 & degenerate<1);

test_data = dat(:,:,:,idx);
testylabels = lab(idx,:);
%% Network
layers = [
    imageInputLayer([720,1,3],'Name','Input')
    
    convolution2dLayer([11,1],30,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer([2,1],'Stride',2,'Name','avpool')
    
    convolution2dLayer([5,1],15,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer([2,1],'Stride',2,'Name','avpool')
    
    convolution2dLayer([3,1],12,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_2')
    
    convolution2dLayer([3,1],12,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_2')
    
%     fullyConnectedLayer(600,'Name','fc1')
%     reluLayer('Name','Relu3')
%     batchNormalizationLayer('Name','BN_2')
%     
%     fullyConnectedLayer(300,'Name','fc1')
%     reluLayer('Name','Relu3')
%     batchNormalizationLayer('Name','BN_2')
    
    
    fullyConnectedLayer(200,'Name','fc1')
    reluLayer('Name','Relu3')
    batchNormalizationLayer('Name','BN_2')
   
    dropoutLayer(0.8)
    
    fullyConnectedLayer(100,'Name','fc2')
    reluLayer('Name','Relu4')
    batchNormalizationLayer('Name','BN_3')
    
%     dropoutLayer(0.7)
    
    fullyConnectedLayer(80,'Name','fc2')
    reluLayer('Name','Relu4')
    batchNormalizationLayer('Name','BN_3')
    
    
%     fullyConnectedLayer(35,'Name','fc2')
%     reluLayer('Name','Relu4')
%     batchNormalizationLayer('Name','BN_3')
    
    fullyConnectedLayer(40,'Name','fc2')
    reluLayer('Name','Relu4')
    batchNormalizationLayer('Name','BN_3')
    
%     fullyConnectedLayer(20,'Name','fc2')
%     reluLayer('Name','Relu4')
%     batchNormalizationLayer('Name','BN_3')
    
    fullyConnectedLayer(9,'Name','fc3')
   
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

[convnet,info] = trainNetwork(input_data,ylabels,layers,options);

%%
save('regression_network.mat','convnet','layers');

%%

YPredicted = predict(convnet,test_data);

bottom_wall_rms = sqrt(sum(sum((YPredicted(:,1) - testylabels(:,1)).^2))/length(testylabels));
top_wall_rms = sqrt(sum(sum((YPredicted(:,2) - testylabels(:,2)).^2))/length(testylabels));
left_wall_rms = sqrt(sum(sum((YPredicted(:,3) - testylabels(:,3)).^2))/length(testylabels));
ceiling_wall_rms = sqrt(sum(sum((YPredicted(:,4) - testylabels(:,4)).^2))/length(testylabels));

light_rms = sqrt(sum(sum((YPredicted(:,5:7) - testylabels(:,5:7)).^2))/length(testylabels(:,5:7)));
cyl1_rms = sqrt(sum(sum((YPredicted(:,8:9) - testylabels(:,8:9)).^2))/length(testylabels(:,8:9)));

disp(['Bottom wall position RMS: ',num2str(bottom_wall_rms)])
disp(['Top wall position RMS: ',num2str(top_wall_rms)])
disp(['Left wall position RMS: ',num2str(left_wall_rms)])
disp(['Ceiling position RMS: ',num2str(ceiling_wall_rms)])
disp(['Light position RMS: ',num2str(light_rms)])
disp(['Cylinder position RMS: ',num2str(cyl1_rms)])

%% Plot lights

skip = 20;

colormap(colorcube)
c = linspace(1,10,length(YPredicted(1:skip:end,5)));

scatter(YPredicted(1:skip:end,5),YPredicted(1:skip:end,7),[],c,'filled')
hold on
scatter(testylabels(1:skip:end,5),testylabels(1:skip:end,7),[],c,'filled')

for ii = 1:skip:length(YPredicted)
    line([YPredicted(ii,5), testylabels(ii,5)],[YPredicted(ii,7),testylabels(ii,7)])
end
axis equal

%% Plot Objects

skip = 20;

colormap(colorcube)
c = linspace(1,10,length(YPredicted(1:skip:end,5)));

scatter(YPredicted(1:skip:end,8),YPredicted(1:skip:end,9),[],c,'filled')
hold on
scatter(testylabels(1:skip:end,8),testylabels(1:skip:end,9),[],c,'filled')

for ii = 1:skip:length(YPredicted)
    line([YPredicted(ii,8), testylabels(ii,8)],[YPredicted(ii,9),testylabels(ii,9)])
end
axis equal

%% Plot Walls

skip = 20;

for ii = 1:skip:length(YPredicted)
    
    lwp = ylabels(ii,3);
    bwp = ylabels(ii,1);
    twp = ylabels(ii,2);

    lp = ylabels(ii,[5,7]);
    cyl = ylabels(ii,8:9);
    
    line([lwp,0],[bwp,bwp],'Color',[0,0,1])
    line([lwp,0],[twp,twp],'Color',[0,0,1])
    line([lwp,lwp],[twp,bwp],'Color',[0,0,1])
    
    lwp = YPredicted(ii,3);
    bwp = YPredicted(ii,1);
    twp = YPredicted(ii,2);

    line([lwp,0],[bwp,bwp],'Color',[1,0,0])
    line([lwp,0],[twp,twp],'Color',[1,0,1])
    line([lwp,lwp],[twp,bwp],'Color',[1,0,0])
end

axis equal

%%

for i = 10:19

    num = i;

    subplot(3,3,i-9)

    lwp = ylabels(num,3);
    bwp = ylabels(num,1);
    twp = ylabels(num,2);

    lp = ylabels(num,[5,7]);
    cyl = ylabels(num,8:9);

    line([lwp,0],[bwp,bwp],'Color',[0,0,1])
    line([lwp,0],[twp,twp],'Color',[0,0,1])
    line([lwp,lwp],[twp,bwp],'Color',[0,0,1])
    hold on
    scatter(lp(1),lp(2),'d','filled','b')
    circle(cyl(1),cyl(2),0.22,[0,0,1])
    hold on
    axis equal
    
    
    lwp = YPredicted(num,3);
    bwp = YPredicted(num,1);
    twp = YPredicted(num,2);

    lp = YPredicted(num,[5,7]);
    
    cyl = YPredicted(num,8:9);

    line([lwp,0],[bwp,bwp],'Color',[1,0,0])
    line([lwp,0],[twp,twp],'Color',[1,0,1])
    line([lwp,lwp],[twp,bwp],'Color',[1,0,0])
    hold on
    scatter(lp(1),lp(2),'d','filled','r')
    circle(cyl(1),cyl(2),0.22,[1,0,0])
    hold on
    axis equal
    xlim([-11,0])
    ylim([-8,8])
end