clc
clear all
close all

[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick an Image File');
I = imread([pathname,filename]);

Istrech = imadjust(I,stretchlim(I));
figure(1),imshow(Istrech)
title('Contrast stretched image')
%K = medfilt2(Istrech);
%figure(3),imshow(K)

%% Convert RGB image to gray
I1 = rgb2gray(Istrech);
figure(2),imshow(I1,[])
title('RGB to gray (contrast stretched) ')
I = imresize(I,[200,200]);

gaussianFilter = fspecial('gaussian',20, 10);
img_filted = imfilter(I1, gaussianFilter,'symmetric');
figure
imshow(img_filted);
title('gaussianFilter Filted Image');
filted_edges = edge(img_filted, 'Canny');
figure();
subplot(121);
imshow(filted_edges);
title('Edges found in filted image');
img_edges = edge(I1, 'Canny');
subplot(122);
imshow(img_edges);
%% Apply median filter to smoothen the image
K = medfilt2(I1);
figure(4),imshow(K)
title('median filter')

%MSE and PSNR measurement
[row, col] = size(I);
mse = sum(sum((I(1,1) - K(1,1)).^2)) / (row * col);
psnr = 10 * log10(255 * 255 / mse);

disp('<--------------- Median  filter  ---------------------------->');
disp('Mean Square Error ');
disp(mse);
disp('Peak Signal to Noise Ratio');
disp(psnr);
disp('<--------------------------------------------------------->');
imgID = 2;
HSV=rgb2hsv(I);
figure,imshow(HSV),title('HSV COLOUR TRANSFORM IMAGE');
% %%SEPARATE THREE CHANNELS%%%
H=HSV(:,:,1);
S=HSV(:,:,2);
V=HSV(:,:,3);

figure,imshow(H),title('H-CHANNEL IMAGE');
figure,imshow(S),title('S-CHANNEL IMAGE');
figure,imshow(V),title('V-CHANNEL IMAGE');

figure,
subplot(1,3,1),imshow(H),title('H-CHANNEL');
subplot(1,3,2),imshow(S),title('S-CHANNEL');
subplot(1,3,3),imshow(V),title('V-CHANNEL');

% %% PERFORM RGB TO GRAY CONVERSION ON THE V-CHANNEL IMAGE%%%%
[m n o]=size(V);
if o==3
    gray=rgb2gray(V);
else
    gray=V;
end
figure,imshow(gray);title('V- CHANNEL GRAY IMAGE');

ad=imadjust(gray);
figure,imshow(ad);title('ADJUSTED GRAY IMAGE');
% 
% %%%%TO PERFORM BINARY CONVERSION ON THE ADJUSTED GRAY IMAGE%%%%%
bw=im2bw(gray,0.5);
figure,imshow(bw);title('BLACK AND WHITE IMAGE');
 
%TAKE COMPLEMENT TO THE BLACK AND WHITE IMAGE %%%%
bw=imcomplement(bw);
figure,imshow(bw);title('COMPLEMENT IMAGE');
 
%TO PERFORM MORPHOLOGICAL OPERATIONS IN THE BW IMAGE%%%%
%FILL HOLES%%
bw=imfill(bw,'holes');
figure,imshow(bw),title('EDGE BASED SEGMENTATION');
%DILATE OPERATION
SE=strel('square',3);
bw=imdilate(bw,SE);
figure,imshow(bw),title('DILATED IMAGE');

fontSize = 10;
	redBand = I(:, :, 1);
	greenBand = I(:, :, 2);
	blueBand = I(:, :, 3);
	% Display them.
	figure
	imshow(redBand);
	title('ENCHANCEMENT_1', 'FontSize', fontSize);
	figure
	imshow(greenBand);
	title('ENCHANCEMENT_2', 'FontSize', fontSize);
	figure
	imshow(blueBand);
	title('ENCHANCEMENT_3', 'FontSize', fontSize);
    
tic;


 signal1 = feature_ext(I);   
%% segmentation
L = kmean(K);

disp('Segmentation.');

%%

cform = makecform('srgb2lab');
% Apply the colorform
lab_he = applycform(I,cform);

% Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 3;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
%[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols);
%figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end


figure,
subplot(3,1,1);imshow(segmented_images{1});title('color based segment');
subplot(3,1,2);imshow(segmented_images{2});title('segmentation');
subplot(3,1,3);imshow(segmented_images{3});title('Cluster 3');
set(gcf, 'Position', get(0,'Screensize'));


Img = double(I(:,:,1));
epsilon = 1;
switch imgID

    case 1
        num_it =1000;
        rad = 8;
        alpha = 0.3;% coefficient of the length term
        mask_init  = zeros(size(Img(:,:,1)));
        mask_init(15:78,32:95) = 1;
        seg = local_AC_MS(Img,mask_init,rad,alpha,num_it,epsilon);
    case 2
        num_it =800;
        rad = 9;
        alpha = 0.003;% coefficient of the length term
        mask_init = zeros(size(Img(:,:,1)));
        mask_init(53:77,56:70) = 1;
        seg = local_AC_UM(Img,mask_init,rad,alpha,num_it,epsilon);
    case 3
        num_it = 1500;
        rad = 5;
        alpha = 0.001;% coefficient of the length term
        mask_init  = zeros(size(Img(:,:,1)));
        mask_init(47:80,86:99) = 1;
        seg = local_AC_UM(Img,mask_init,rad,alpha,num_it,epsilon);
end


[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');

DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);

g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
    
feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness];
disp('-----------------------------------------------------------------');
disp('Contrast = ');
disp(Contrast);
disp('Correlation = ');
disp(Correlation);
disp('Energy = ');
disp(Energy);
disp('Mean = ');
disp(Mean);
disp('Standard_Deviation = ');
disp(Standard_Deviation);
disp('Entropy = ');
disp(Entropy);
disp('RMS = ');
disp(RMS);
disp('Variance = ');
disp(Variance);
disp('Kurtosis = ');
disp(Kurtosis);
disp('Skewness = ');
disp(Skewness);
load Trainset.mat
 xdata = meas;
 group = label;
acc=accuracy_image(feat); 
disp(acc);
addpath('../../Training' , '../../mdCNN' , '../../utilCode' );

numTest=60;numTrain=240;
x=randi(16,1,numTrain+numTest)-1; 
xBin=[mod(x,2) ;mod(floor(x/2),2) ;mod(floor(x/4),2) ;mod(floor(x/8),2)]; 

samples = [repmat((1:10)',1,size(xBin,2)) ; xBin ; rand(10,size(xBin,2))/2*mean(xBin(:))]; 

dataset=[];
for idx=1:size(samples,2)
    if (idx>numTrain)
        dataset.I_test{idx-numTrain} = samples(:,idx-numTrain);
        dataset.labels_test(idx-numTrain)=x(idx-numTrain);
    else
    dataset.I{idx} = samples(:,idx);
    dataset.labels(idx)=x(idx);
    end
end

net = CreateNet('../../Configs/1d.conf');  % small 1d fully connected net,will converge faster

net   =  Train(dataset,net, 100);

checkNetwork(net,Inf,dataset,1);
% help dialogue box classification of disease
result = cnn_classifier(feat,meas,label);
helpdlg(result);

load('Accuracy_Data.mat')
Accuracy_Percent= zeros(200,1);
for i = 1:800
data = Train_Feat;
groups = ismember(Train_Label,1);
[train,feat] = crossvalind('HoldOut',groups);
cp = classperf(groups);
 classperf(cp,feat);
Accuracy = cp.CorrectRate*2;
Accuracy_Percent(i) = Accuracy.*100;
end
Max_Accuracy = max(Accuracy_Percent);
sprintf('Accuracy of cnn with 800 iterations is: %g%%',Max_Accuracy)

	fontSize = 10;
	
	% Compute and plot the red histogram.
	hR = figure
	[countsR, grayLevelsR] = imhist(redBand);
	maxGLValueR = find(countsR > 0, 1, 'last');
	maxCountR = max(countsR);
	bar(countsR, 'r');
	grid on;
	xlabel('GRAY VALUE');
	ylabel('PIXEL');
	title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the green histogram.
	hG = figure
	[countsG, grayLevelsG] = imhist(greenBand);
	maxGLValueG = find(countsG > 0, 1, 'last');
	maxCountG = max(countsG);
	bar(countsG, 'g', 'BarWidth', 0.95);
	grid on;
	xlabel('GRAY VALUE');
	ylabel('PIXEL');
    title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the blue histogram.
	hB = figure
	[countsB, grayLevelsB] = imhist(blueBand);
	maxGLValueB = find(countsB > 0, 1, 'last');
	maxCountB = max(countsB);
	bar(countsB, 'b');
	grid on;
	xlabel('GRAY VALUE');
	ylabel('PIXEL');
	title('CNN_GRAPH', 'FontSize', fontSize);

	maxGL = max([maxGLValueR,  maxGLValueG, maxGLValueB]);

	maxCount = max([maxCountR,  maxCountG, maxCountB]);
	
	% Plot all 3 histograms in one plot.
	figure
	plot(grayLevelsR, countsR, 'r', 'LineWidth', 2);
	grid on;
	xlabel('Gray Levels');
	ylabel('Pixel Count');
	hold on;
	plot(grayLevelsG, countsG, 'g', 'LineWidth', 2);
	plot(grayLevelsB, countsB, 'b', 'LineWidth', 2);
	title('ALL OVER GASTRO REGION', 'FontSize', fontSize);

