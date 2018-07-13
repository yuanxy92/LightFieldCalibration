function [source_features,target_features] = SURF(I1,I2)

    grayI1 = rgb2gray(I1);
    grayI2 = rgb2gray(I2);

    points1 = detectSURFFeatures(grayI1);
    points2 = detectSURFFeatures(grayI2);

    [f1, vpts1] = extractFeatures(grayI1, points1);
    [f2, vpts2] = extractFeatures(grayI2, points2);

    index_pairs = matchFeatures(f1, f2) ;
    matched_pts1 = vpts1(index_pairs(:, 1));
    matched_pts2 = vpts2(index_pairs(:, 2));

    [n,~] = size(matched_pts1);

    source_features = zeros(n,2);
    target_features = zeros(n,2);

    for i=1:n
        source_features(i,:) = matched_pts1(i).Location;
        target_features(i,:) = matched_pts2(i).Location;
    end
    
    [~,inliners] = EstimateHomographyByRANSAC(source_features',target_features', 0.001);
    source_features = source_features(inliners,:);
    target_features = target_features(inliners,:);

    
    
    
end