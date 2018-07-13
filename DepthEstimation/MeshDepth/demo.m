%Partial Matlab code for "Bundled Camera Paths for Video Stabilization" (SIGGRAPH 2013)
%Implementation of motion model estimation.
%1. As-similar-as-possible warping.
%2. Local homography estimation on mesh quads.
%require vision tool box for detectSURFFeatures, or you may want to use
%your own features. (N x 2)


clear all;
clc;

addpath('mesh');
addpath('RANSAC');

I1 = imread('examples/3/s.png');
I2 = imread('examples/3/t.png');
fprintf('detect surf features...');
[I1_features,I2_features]=SURF(I1,I2);
fprintf('[DONE]');

if length(I1_features) < 20
    error('not enough matched features');
    return;
end

[height,width,~] = size(I1);
%3x3 mesh
quadWidth = width/(2^3);
quadHeight = height/(2^3);

% %4x4 mesh
% quadWidth = width/(2^4);
% quadHeight = height/(2^4);

lamda = 1; %mesh more rigid if larger value. [0.2~5]
asap = AsSimilarAsPossibleWarping(height,width,quadWidth,quadHeight,lamda);
asap.SetControlPts(I1_features,I2_features);%set matched features
asap.Solve();            %solve Ax=b for as similar as possible
homos = asap.CalcHomos();% calc local hommograph transform

gap = 50;
I1warp = asap.Warp(I1,gap);                     %warp source image to target image
I1warpmesh = asap.destin.drawMesh(I1warp,gap);  %draw mesh on the warped source image
imshow(I1warpmesh);

gap = 0;
I1warp = asap.Warp(I1,gap);                     
imwrite(I1warp,'examples/3/warp.png');

%access local homography
[h,w,~,~] = size(homos);
for i=1:h-1
    for j=1:w-1
       H(:,:) = homos(i,j,:,:);
       fprintf('Quad=[%d %d]\n',i,j);
       H
    end
end
