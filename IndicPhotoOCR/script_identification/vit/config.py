common_config={
    'pretrained_vit_model': 'google/vit-base-patch16-224-in21k'
}

train_config = {
    'epochs': 20,
    'max_images_real':1900,
    'classes':12,
    'hindi_path_real': '<path_for_hindi_dataset>',
    'english_path_real':'<path_for_eng_dataset>',
    'gujarati_path_real':'<path_for_gujarati_dataset>',
    'punjabi_path_real':'<path_for_punjabi_dataset>',
    'assamese_path_real':'<path_for_assamese_dataset>',
    'bengali_path_real':'<path_for_bengali_dataset>',
    'kannada_path_real':'<path_for_kannada_dataset>',
    'malayalam_path_real':'<path_for_malayalam_dataset>',
    'marathi_path_real':'<path_for_marathi_dataset>',
    'odia_path_real':'<path_for_odia_dataset>',
    'tamil_path_real':'<path_for_tamil_dataset>',
    'telugu_path_real':'<path_for_telegu_dataset>',
    'checkpoints_dir': '<path_for_model>'
    
}
train_config.update(common_config)

test_config = {
    'reload_model': '<path_for_model>',
    'max_images':2000,
    'classes':12,
    'hindi_path_real': '<path_for_hindi_dataset>',
    'english_path_real':'<path_for_eng_dataset>',
    'gujarati_path_real':'<path_for_gujarati_dataset>',
    'punjabi_path_real':'<path_for_punjabi_dataset>',
    'assamese_path_real':'<path_for_assamese_dataset>',
    'bengali_path_real':'<path_for_bengali_dataset>',
    'kannada_path_real':'<path_for_kannada_dataset>',
    'malayalam_path_real':'<path_for_malayalam_dataset>',
    'marathi_path_real':'<path_for_marathi_dataset>',
    'odia_path_real':'<path_for_odia_dataset>',
    'tamil_path_real':'<path_for_tamil_dataset>',
    'telugu_path_real':'<path_for_telegu_dataset>',
    

}
test_config.update(common_config)



infer_config = {
    'model_path':'<path_for_model>',
    'img_path': 'image_path',
    'folder_path':'<path_dataset_folder>',
    'csv_path':'<csv_path>',
}


infer_config.update(common_config)

