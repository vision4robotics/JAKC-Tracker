function seq = load_video_information(type) % UAV123_10fps

%     type =  'UAV123_10fps';
    database_folder = '.\UAV123_10fps\data_seq';
    ground_truth_folder = '.\UAV123_10fps\anno';
    video_name = choose_video_UAV(ground_truth_folder);
    seq = load_video_info_UAV123(video_name, database_folder, ground_truth_folder, type);
    seq.video_name = video_name;