args = {
    
    # data path
    'train_path' : '/home/tako/mesrwi/Pose2Muscle/datasplit/train_split.txt',
    'valid_path' : '/home/tako/mesrwi/Pose2Muscle/datasplit/eval_split.txt',
    'test_path' : '/home/tako/mesrwi/Pose2Muscle/datasplit/eval_split.txt',
    
    # training
    'epoch' : 300,
    'device' : 'cuda:2', # 'cuda:1',
    'accelerator' : 'gpu',
    'devices' : 2,
    'batch_size' : 64, # 32
    'num_workers' : 8,
    'seed' : 42,
    
    # scheduler & optimizer
    'mode' : 'max',
    'min_lr' : 1.0e-07,
    'factor' : 0.99,
    'patience' : 8,
    'beam_size' : 1,
    'lr' : 0.00005,
    'betas' : (0.9, 0.998),
    'weight_decay' : 0.001,
    
    # loss
    # 'recognition_loss_weight' : 5.0,
    # 'translation_loss_weight' : 1.0,
    
    # 초기화
    "initializer": "xavier",
    "bias_initializer": "zeros",
    "init_gain": 1.0,
    "embed_initializer": "xavier",
    "embed_init_gain": 1.0,
    "tied_softmax": False,
    
    # wandb
    'project_name' : 'Pose2Muscle',
    'task_name' : 'GTM_one-target:ES',
    
    # model
    'model_path' : './checkpoint',

    # input data
    'video_feat_dim' : 1024,
    
    # model - pose_embedding
    'pose_embedding_dim' : 256,
    'pose_embedding_activation' : 'relu',
    
    # model - subject_embedding
    'subject_embedding_dim' : 32,
    'subject_embedding_activation' : 'relu',

    # model - encoder
    'encoder_dim' : 256,
    'encoder_nhead' : 4,
    'encoder_layer' : 2,
    'encoder_ff_size' : 2048,
    'encdoer_emb_dropout' : 0.1,
    
    # model - decoder
    'decoder_dim' : 512,
    'decoder_nhead' : 8,
    'decoder_layer' :3,
    'decoder_ff_size' : 2048,
    'decoder_emb_dropout' : 0.1,
    
    # model - regression head
    'target_muscle' : ['ES'] # ['ES', 'UT', 'BB', 'FDS', 'ED', 'BF', 'VL']
}