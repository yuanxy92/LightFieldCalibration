clear;
close all;
fclose all;
addpath(genpath('MeshDepth'));

% depth = single(imread('E:\Project\LightFieldGiga\data\data1\result_fusion\final\panorama_depth2.png'));
% img = imread('E:\Project\LightFieldGiga\data\data1\result_fusion\final\mesh_texture2.png');

depth = single(imread('E:\Project\LightFieldGiga\data\depth_test\depth_c.png'));
img = imread('E:\Project\LightFieldGiga\data\depth_test\test.jpg');

disparity = 1 ./ depth * 100000;

[rows, cols] = size(depth);



ptrows = floor(rows / 20);
ptcols = floor(cols / 20);

pts_src = zeros(ptrows * ptcols, 2);
pts_dst = zeros(ptrows * ptcols, 2);
ind = 1;
for i = 1:ptrows
    for j = 1:ptcols
        pts_src(ind, :) = [j * 20 - 10, i * 20 - 10];
        pts_dst(ind, :) = [pts_src(ind, 1) + disparity(pts_src(ind, 2), ...
            pts_src(ind, 1)), pts_src(ind, 2)];
        ind = ind + 1;
    end
end

quadWidth = cols/(100);
quadHeight = rows/(80);
lamda = 1.2; %mesh more rigid if larger value. [0.2~5]
asap = AsSimilarAsPossibleWarping(rows, cols, quadWidth, quadHeight, lamda);
asap.SetControlPts(pts_src, pts_dst);%set matched features
asap.Solve();            %solve Ax=b for as similar as possible
homos = asap.CalcHomos();% calc local hommograph transform

dst = asap.destin.xMat;
src = asap.source.xMat;
disparity2 = imresize(dst - src, [rows, cols]);
depth2 = uint16(1 ./ disparity2 * 100000);
imwrite(depth2, 'E:\Project\LightFieldGiga\data\depth_test\depth_smooth.png');

