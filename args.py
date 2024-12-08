args = {
    
    # data path
    'train_path' : '/home/tako/SSL/class_AI/DATA/full_frame/phoenix14t.pami0.train',
    'valid_path' : '/home/tako/SSL/class_AI/DATA/full_frame/phoenix14t.pami0.dev',
    'test_path' : '/home/tako/SSL/class_AI/DATA/full_frame/phoenix14t.pami0.test',
    
    # training
    'epoch' : 100,
    'device' : 'cuda:2', # 'cuda:1',
    'accelerator' : 'gpu',
    'devices' : 2,
    'batch_size' : 4, # 32
    'num_workers' : 4,
    'seed' : 42,
    
    # scheduler & optimizer
    'mode' : 'max',
    'min_lr' : 1.0e-07,
    'factor' : 0.99,
    'patience' : 8,
    'beam_size' : 1,
    'lr' : 1e-3,
    'betas' : (0.9, 0.998),
    'weight_decay' : 0.001,
    
    # loss
    'recognition_loss_weight' : 5.0,
    'translation_loss_weight' : 1.0,
    
    # 초기화
    "initializer": "xavier",
    "bias_initializer": "zeros",
    "init_gain": 1.0,
    "embed_initializer": "xavier",
    "embed_init_gain": 1.0,
    "tied_softmax": False,
    
    # wandb
    'project_name' : 'Pose2Muscle',
    'task_name' : 'testing',
    
    # model
    'model_path' : './model_weight/testing',

    # input data
    'video_feat_dim' : 1024,
    
    # model - pose_embedding
    'pose_embedding_dim' : 512,
    'pose_embedding_activation' : 'relu',

    # model - encoder
    'encoder_dim' : 512,
    'encoder_nhead' : 8,
    'encoder_layer' :3,
    'encoder_ff_size' : 2048,
    'encdoer_emb_dropout' : 0.1,
    
    # model - decoder
    'decoder_dim' : 512,
    'decoder_nhead' : 8,
    'decoder_layer' :3,
    'decoder_ff_size' : 2048,
    'decoder_emb_dropout' : 0.1,
}