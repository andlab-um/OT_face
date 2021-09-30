

myCluster = parcluster('local');
myCluster.NumWorkers = 12;
parpool(12);

result_pre = '/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/mvpa_roi/result/perm';
prefix = '/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/fmri_data/fMRI_haiyan_processed';

subj_list = [105;106;107;108;113;114;129;130;131;132;137;138;153;154;155;156;161;162;119;120;125;126;127;128;143;144;149;150;151;152;115;116;121;122;123;124;139;140;145;146;147;148;109;110;111;112;117;118;133;134;135;136;141;142;157;158;159;160;165;166];
cond_list = convertCharsToStrings({'OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','OT','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl','Pl'});

nsub = 0;
for i = 1:length(subj_list)
    sub = strcat('s',num2str(subj_list(i)));
    subpath = fullfile(prefix,sub);
    if  (isfolder(subpath))
        nsub = nsub+1;
    end
end

nbeta = nsub * 2;
filenames = cell(nbeta,1);

chunk = [];
label = [];

j=1;
k=1;

for i = 1:length(subj_list)
    sub = strcat('s',num2str(subj_list(i)));
    subpath = fullfile(prefix,sub);
    
    if (cond_list(i) == "OT") && (isfolder(subpath))

% chunk: data from one subject
% see tutorial
% label: 1 for self, 0 for other

        filenames{j,1} = fullfile(subpath,'beta_0001.img');
        j = j+1;
        chunk = [chunk,k];
        label = [label,1]
        
        filenames{j,1} = fullfile(subpath,'beta_0003.img');
        j = j+1;
        chunk = [chunk,k];
        label = [label,1];
        
        filenames{j,1} = fullfile(subpath,'beta_0011.img');
        j = j+1;
        chunk = [chunk,k];
        label = [label,1];

        filenames{j,1} = fullfile(subpath,'beta_0013.img');
        j = j+1;
        chunk = [chunk,k];
        label = [label,1];
        
        filenames{j,1} = fullfile(subpath,'beta_0002.img');
        j = j+1;
        chunk = [chunk,k];
        label = [label,0]
        
        filenames{j,1} = fullfile(subpath,'beta_0004.img');
        j = j+1;
        chunk = [chunk,k];
        label = [label,0];
        
        filenames{j,1} = fullfile(subpath,'beta_0012.img');
        j = j+1;
        chunk = [chunk,k];
        label = [label,0];

        filenames{j,1} = fullfile(subpath,'beta_0014.img');
        j = j+1;
        chunk = [chunk,k];
        label = [label,0];
        
        k = k+1;
    end

    
end

parfor i = 1:120
    n = length(label)
    result_dir = strcat(result_pre,num2str(i),'/');
    
    randseq = randi([1, n],n,1);
    
    t_filenames = reshape(filenames(randseq),[],1);
	t_chunk = reshape(chunk(randseq),[],1);
    t_label = reshape(label(randseq),[],1);
    
    TDTmvpa(t_filenames, t_chunk, t_label, result_dir);
    
    seqfilename = fullfile(result_dir, 'lableSeq.mat')
    parsave(seqfilename, t_filenames, t_chunk, t_label);
end



