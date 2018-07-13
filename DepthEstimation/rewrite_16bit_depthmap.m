clear;
close all;
fclose all;

for i = 10:13
   img = imread(sprintf('E:\\Project\\LightFieldGiga\\data\\result\\%04d_depth_refine.png', i));
   imwrite(img, sprintf('E:\\Project\\LightFieldGiga\\data\\result\\%04d_depth_refine.png', i));
end