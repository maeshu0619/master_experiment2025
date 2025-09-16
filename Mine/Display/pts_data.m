clear;clc;close all
%% 読み込み
xyzirgb = readmatrix("F:\Dataset\PartAnnotation\02691156\points\1a04e3eab45ca15dd86060f189eb133.pts");

%% point cloud変数を作成
pt = pointCloud(xyzirgb(:,1:3),"Color",xyzirgb(:,5:7)/255,'Intensity',xyzirgb(:,4));

%% RGB表示
figure;pcshow(pt);title('RGB表示')

%% intensity表示
figure; pcshow(pt.Location,pt.Intensity);title('intensity表示');hold on
colorbar