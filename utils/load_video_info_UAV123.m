% Loads relevant information of UAV123 in the given path.
function seq = load_video_info_UAV123(video_name, database_folder, ground_truth_path, type)

    seqs = configSeqs(database_folder, type);
    
    i=1;
    while ~strcmpi(seqs{i}.name,video_name)
            i=i+1;
    end
    
    seq.video_name = seqs{i}.name;         
    seq.name = seqs{i}.name;
    seq.video_path = seqs{i}.path;  
    seq.st_frame = seqs{i}.startFrame;      
    seq.en_frame = seqs{i}.endFrame;     
    seq.len = seq.en_frame-seq.st_frame+1;  
    
    ground_truth = dlmread([ground_truth_path '\' seq.video_name '.txt']);
    seq.ground_truth = ground_truth;       
    
    seq.init_rect = ground_truth(1,:);     
    target_sz = [ground_truth(1,4), ground_truth(1,3)];
    seq.target_sz = target_sz;
	seq.pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
    
    img_path = seq.video_path;
    img_files_struct = dir(fullfile(img_path, '*.jpg'));
    img_files = {img_files_struct.name};                     
    seq.img_files = img_files;
    seq.s_frames = img_files(1, seq.st_frame : seq.en_frame); 
    for i = 1 : length(seq.s_frames)
        seq.s_frames{i} = [img_path seq.s_frames{i}];        
    end
    