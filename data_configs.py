import os


nlpcda_data_path = 'D:/Apps/Anaconda3/envs/general-torch/Lib/site-packages/nlpcda/data'
entities_file = os.path.join(nlpcda_data_path, "entities.txt")
logic_pairs_file = os.path.join(nlpcda_data_path, 'logic_pairs.txt')

DATA_AUG_CONFIGS = {
    'random_entity':{
        'base_file':entities_file, 
        'create_num':2, 
        'change_rate':0.2, 
        'seed':1024, 
        'prop':0.4,  
    }, 
    'random_delete_char':{
        'create_num':2, 
        'change_rate':0.05, 
        'seed':1024, 
        'prop':0.4, 
    }, 
    'random_swap':{
        'create_num':2, 
        'change_rate':0.2, 
        'seed':1024, 
        'prop':0.4, 
    }, 
    'random_swap_order':{
        'create_num':2,
        'char_gram':5,  
        'change_rate':0.2, 
        'seed':1024, 
        'prop':0.4, 
    }, 
    'random_add_similar':{
        'create_num':2,
        'change_rate':0.2, 
        'seed':1024, 
        'prop':0.4, 
    }, 
    'random_swap_logic_words':{
        'base_file':logic_pairs_file, 
        'create_num':2,
        'change_rate':0.5, 
        'seed':1024, 
        'prop':0.75, 
    }, 
    'random_swap_logic_order':{
        'create_num':2,
        'change_rate':1, 
        'seed':1024, 
        'prop':0.75, 
    }, 
    'random_replace_antonym':{
        'create_num':2,
        'change_rate':0.2, 
        'seed':1024, 
        'prop':0.4, 
    }, 
}

def get_dataset_configs(args, device, mlm_finetune=False):    
    oob_model_name = None if mlm_finetune else args.oob_model_name
    da_configs = None if mlm_finetune else DATA_AUG_CONFIGS

    TRAIN_DATASET_CONFIGS = {
        'model_name':args.model_name,
        'aux_model_name':oob_model_name, 
        'maxlength':args.maxlength,
        'train_val_split':-1,
        'test':False, 
        'split_words':args.split_words,
        'remove_username':args.remove_username,
        'remove_punctuation':args.remove_punctuation, 
        'to_simplified':args.to_simplified, 
        'emoji_to_text':args.emoji_to_text,
        'device':device, 
        'split_words':False, 
        'cut_all':False, 
        'da_configs':da_configs, 
    }

    DEV_DATASET_CONFIGS = {
        'model_name':args.model_name,
        'aux_model_name':oob_model_name, 
        'maxlength':args.maxlength,
        'train_val_split':-1,
        'test':False, 
        'split_words':args.split_words,
        'remove_username':args.remove_username,
        'remove_punctuation':args.remove_punctuation, 
        'to_simplified':args.to_simplified, 
        'emoji_to_text':args.emoji_to_text,
        'device':device, 
        'split_words':False, 
        'cut_all':False, 
    }

    TEST_DATASET_CONFIGS = {
        'model_name':args.model_name,
        'aux_model_name':oob_model_name, 
        'maxlength':args.maxlength,
        'train_val_split':-1,
        'test':True, 
        'split_words':args.split_words,
        'remove_username':args.remove_username,
        'remove_punctuation':args.remove_punctuation, 
        'to_simplified':args.to_simplified, 
        'emoji_to_text':args.emoji_to_text,
        'device':device, 
        'split_words':False, 
        'cut_all':False, 
    }

    return TRAIN_DATASET_CONFIGS, DEV_DATASET_CONFIGS, TEST_DATASET_CONFIGS
