function [matching_pts, valid_index] = remove_outliers(matching_points_left, matching_points_right)
%% function to remove outliers
%
num = 1;
valid_index = [];
matching_pts = zeros(size(matching_points_left, 1), 4);
for i = 1:size(matching_points_left, 1)
    pt_left = matching_points_left(i).Location;
    pt_right = matching_points_right(i).Location;
    if abs(pt_left(1) - pt_right(1)) <= 30 &&  abs(pt_left(2) - pt_right(2)) <= 4
        matching_pts(num, :) = [pt_left, pt_right];
        num = num + 1;
        valid_index = [valid_index, i];
    end
end
matching_pts = matching_pts(1:(num - 1), :);