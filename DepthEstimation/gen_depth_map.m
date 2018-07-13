clear;
close all;
fclose all;

videoname1 = '..\\data\\raw\\5\\cam_00.avi';
videoname2 = '..\\data\\raw\\5\\cam_01.avi';
imgname1 = '..\\data\\0000_left.jpg';
imgname2 = '..\\data\\0000_right.jpg';
depthname = '..\\data\\0000_depth.png';
matchname = 'match.tmp';
v1 = VideoReader(videoname1);
v2 = VideoReader(videoname2);
addpath(genpath('Interpolator'));

ind = 0;
while hasFrame(v1)
    if ind > 500
        break;
    end
    
    fprintf('Process video frame index %d ...\n', ind);
    
    for k = 1:2
        img1 = readFrame(v1);
        img2 = readFrame(v2);
    end
    
    %img1 = imresize(img1, 0.5);
    %img2 = imresize(img2, 0.5);
    
    imwrite(img1, imgname1);
    imwrite(img2, imgname2);
    
    % generate edges
    load('.\\Interpolator\\model\\modelFinal.mat');
    edges = edgesDetect(imread(imgname1), model);
    edge_path = sprintf('edges.tmp');
    fid = fopen(edge_path,'wb');
    fwrite(fid,transpose(edges),'single');
    fclose(fid);

    % construct feature
    % feature_construction(imgname1, imgname2, matchname);
    command = sprintf('%s %s %s %d %d %s %s', ...
        'E:\\Project\\LightFieldGiga\\SparseToDenseStereo\\build\\bin\\Release\\SparseFeatureMatch.exe', ...
        strrep(imgname1, '\\', '/'), strrep(imgname2, '\\', '/'), ...
        0, 90, 'match.tmp', sprintf('..\\data\\result\\%04d_confidence.png', ind));
    system(command);

    % run epic flow
    command = sprintf('run_epicflow.bat %s %s %s %s %s', ...
        strrep(imgname1, '\\', '/'), strrep(imgname2, '\\', '/'), ...
        'edges.tmp', 'match.tmp', 'flow.flo');
    system(command);

    % read result and write to depth map
    baseline = 35;
    focal = 7300;
    flow = readFlowFile('flow.flo');
    disparity = flow(:, :, 1);
    depth = -baseline * focal ./ disparity;
    
    imwrite(uint16(depth), sprintf('..\\data\\result\\%04d_depth.png', ind));
    imwrite(img1, sprintf('..\\data\\result\\%04d_left.jpg', ind));
    imwrite(img2, sprintf('..\\data\\result\\%04d_right.jpg', ind));
    
    ind = ind + 1;
end
