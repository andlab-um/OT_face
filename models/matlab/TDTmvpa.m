
function TDTmvpa(filenames, chunk, label, result_dir)

    cfg = decoding_defaults;
    cfg.analysis = 'searchlight';
    cfg.results.dir = result_dir;

    cfg.files.name  = filenames;
    cfg.files.chunk = chunk;
    cfg.files.label = label;

%     cfg.files.mask = '~/Experiments/OTface/data/fmri/mask_all.nii';
    cfg.files.mask = '/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/ROI masks/self_referential.nii';

    cfg.searchlight.unit = 'mm';
    cfg.searchlight.radius = 4;
    cfg.searchlight.spherical = 1;
    cfg.verbose = 2;
    cfg.decoding.train.classification.model_parameters = '-s 0 -t 0 -c 1 -b 0 -q'; 

    cfg.plot_selected_voxels = 10;
    cfg.design = make_design_cv(cfg);

    cfg.decoding.method = 'classification_kernel';
    cfg.design.unbalanced_data = 'ok';
    cfg.basic_checks.DoubleFilenameEntriesOk = 1;

    [results,cfg] = decoding(cfg);
end

