function feature_construction(imgname1, imgname2, matchname)
%% calculate sparse features
img_left = imread(imgname1);
img_right = imread(imgname2);
% detect keypoints
points_left = detectSURFFeatures(rgb2gray(img_left), 'NumOctaves', 4, ...
    'MetricThreshold', 200, 'NumScaleLevels', 10);
points_right = detectSURFFeatures(rgb2gray(img_right), 'NumOctaves', 4, ...
    'MetricThreshold', 200, 'NumScaleLevels', 10);
% extract surf features
[features_left, valid_points_left] = extractFeatures(rgb2gray(img_left), points_left);
[features_right, valid_points_right] = extractFeatures(rgb2gray(img_right), points_right);
% match features
indexPairs = matchFeatures(features_left, features_right, 'Unique', true);
matchedPoints_left = valid_points_left(indexPairs(:, 1), :);
matchedPoints_right = valid_points_right(indexPairs(:, 2), :);
% remove outliers
[matching_pts, valid_index] = remove_outliers(matchedPoints_left, matchedPoints_right);
% matchedPoints_left = matchedPoints_left(valid_index, :);
% matchedPoints_right = matchedPoints_right(valid_index, :);
% save
save_matching_points(matching_pts, matchname);

% figure; showMatchedFeatures(img_left, img_right, matchedPoints_left, matchedPoints_right);

a = 10;
a = a + 1;