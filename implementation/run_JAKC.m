function results = run_JACT(seq)

    
params.hog_cell_size = 4;
params.fixed_area = 200^2;                     % standard area to which we resize the target
params.n_bins = 2^5;                           % number of bins for the color histograms (bg and fg models)
params.learning_rate_pwp = 0.02;               % bg and fg color models learning rate 
params.lambda_scale = 1e-4;                     % regularization weight
params.scale_sigma_factor = 0.5;
params.scale_sigma = 0.1;
params.merge_factor = 0.3;

% fixed setup
params.hog_scale_cell_size = 4;
params.scale_model_factor = 1.0;

params.feature_type = 'fhog';
params.scale_adaptation = true;
params.grayscale_sequence = false; % suppose that sequence is colour

params.img_files = seq.s_frames;
params.img_path = '';
    
s_frames = seq.s_frames;
params.s_frames = s_frames;
params.video_path = seq.video_path;
im = imread([s_frames{1}]);
if(size(im,3)==1)
    params.grayscale_sequence = true;
end

region = seq.init_rect;
x = region(1);
y = region(2);
w = region(3);
h = region(4);
cx = x+w/2;
cy = y+h/2;

% init_pos is the centre of the initial bounding box
params.init_pos = [cy cx];
params.target_sz = round([h w]);

params.inner_padding = 0.2;% defines inner area used to sample colors from the foreground

[params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);

% Deep feature parameters
params.indLayers = [37, 28, 19]; % The CNN layers Conv3-4 in VGG Net 
deep_params.nDim = [512, 512, 256];
deep_params.layers = params.indLayers;
Feat1 = 'conv3'; 
Feat2 = 'conv5'; % conv3, conv4, conv5
switch Feat1
    case 'conv3'
        params.layerInd{1} = 3;
    case 'conv4'
        params.layerInd{1} = 2;
    case 'conv5'
        params.layerInd{1} = 1;
end

switch Feat2
    case 'conv3'
        params.layerInd{2} = 3;
    case 'conv4'
        params.layerInd{2} = 2;
    case 'conv5'
        params.layerInd{2} = 1;
end

params.feat_type = {Feat1, Feat2};

params.t_features = {struct('getFeature_deep',@get_deep,...
                            'deep_params',deep_params)...
                             };
params.t_global.cell_size = 4;
params.t_global.cell_selection_thresh = 0.75^2;

params.lambda1 = 1e-4;                   
params.lambda2 = 0.01;                  
params.gamma = 1;                       
params.output_sigma_factor = {1/32, 1/16};    % label function

params.kernel_type = 'linear'; % 'gaussian' 'polynomial' 'linear'
% gaus kernel
params.gaus1 = 0.5;
params.gaus2 = 0.5;
% poly kernel
params.polya1 = 1;
params.polyb1 = 3;
params.polya2 = 1;
params.polyb2 = 3;

params.tran_sigma = {0.5, 0.5};         
params.polya = {1,1};            
params.polyb = {3,3};             

params.bgl_interv = 20;             

params.learning_rate_cf = 0.01;
params.num_scales = 33;
params.scale_step = 1.03;
params.scale_model_max_area = 32*16; 
params.learning_rate_scale = 0.025;
 
params.visualization = 1;
params.visualization_dbg = 0;

% in runTracker we do not output anything because it is just for debug
if params.visualization_dbg == 1
    params.fout = 1;
else
    params.fout = 0;
end

% start the actual tracking
results = tracker(params, im, bg_area, fg_area, area_resize_factor);