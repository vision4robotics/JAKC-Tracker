function feature_pixels = get_features(image, features, gparams, feat, layerInd)

if ~ iscell(features)
    features = {features};
end

[im_height, im_width, ~, num_images] = size(image);

switch feat
    case 'conv3'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.deep_params.nDim(layerInd), num_images, 'single');
        feature_pixels(:,:,1:features{1}.deep_params.nDim(layerInd),:) = features{1}.getFeature_deep(image,features{1}.deep_params,gparams,layerInd);
    case 'conv5'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.deep_params.nDim(layerInd), num_images, 'single');
        feature_pixels(:,:,1:features{1}.deep_params.nDim(layerInd),:) = features{1}.getFeature_deep(image,features{1}.deep_params,gparams,layerInd);
end
end