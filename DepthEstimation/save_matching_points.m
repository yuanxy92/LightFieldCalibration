function save_matching_points(matching_pts, filename)
%% function to remove outliers
%
fp = fopen(filename, 'w');
for i = 1:size(matching_pts, 1)
   fprintf(fp, '%d\t%d\t%d\t%d\t%d\t%d\n', matching_pts(i, 1), ...
       matching_pts(i, 2), matching_pts(i, 3), matching_pts(i, 4),...
       5.0, 5.0); 
end
fclose(fp);

end