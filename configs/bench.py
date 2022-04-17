import os

config = {}

config['experiment'] = {}
config['experiment']['name'] = 'bench'
config['experiment']['dir'] = os.path.join('experiments', config['experiment']['name'])
config['experiment']['ckpt_save_dir'] = os.path.join(config['experiment']['dir'], 'ckpt')
config['experiment']['recon_latent_codes_dir'] = os.path.join(config['experiment']['dir'], 'recon_latent_codes')
config['experiment']['recon_meshes_dir'] = os.path.join(config['experiment']['dir'], 'recon_meshes')
config['experiment']['eval_results_save_path'] = os.path.join(config['experiment']['dir'], 'eval_results.csv')

config['network'] = {}
config['network']['code_cloud'] = {}
config['network']['code_cloud']['num_codes'] = 1376
config['network']['code_cloud']['code_dim'] = 32
config['network']['code_regularization_lambda'] = 0.0
config['network']['code_position_lambda'] = 3e3

config['train_dataset'] = {}
config['train_dataset']['data_root'] = '...' # To be specified
config['train_dataset']['split'] = 'train_val_02828884'
config['train_dataset']['load_in_memory'] = True
config['train_dataset']['num_samples_per_step'] = 4096
config['train_dataset']['batch_size'] = 16
config['train_dataset']['shuffle'] = True
config['train_dataset']['drop_last'] = True
config['train_dataset']['num_workers'] = 8

config['recon_dataset'] = {}
config['recon_dataset']['data_root'] = '...' # To be specified
config['recon_dataset']['split'] = 'test_02828884'
config['recon_dataset']['load_in_memory'] = True
config['recon_dataset']['num_samples_per_step'] = 4096
config['recon_dataset']['batch_size'] = 1
config['recon_dataset']['shuffle'] = False
config['recon_dataset']['drop_last'] = False
config['recon_dataset']['num_workers'] = 1

config['train'] = {}
config['train']['num_epoch'] = 500
config['train']['decoder_init_lr'] = 1e-3
config['train']['decoder_lr_decay_step'] = 200
config['train']['decoder_lr_decay_rate'] = 0.5
config['train']['latent_codes_init_lr'] = 3e-3
config['train']['latent_codes_lr_decay_step'] = 200
config['train']['latent_codes_lr_decay_rate'] = 0.5

config['reconstruct'] = {}
config['reconstruct']['num_iteration'] = 500
config['reconstruct']['init_lr'] = 1e-2
config['reconstruct']['lr_decay_step'] = 200
config['reconstruct']['lr_decay_rate'] = 0.5
config['reconstruct']['marching_cubes_resolution'] = 128
config['reconstruct']['ckpt_decoder'] = os.path.join(config['experiment']['ckpt_save_dir'], 'decoder.pkl')

config['evaluate'] = {}
config['evaluate']['split_file'] = '.../test_02828884_split.txt' # To be specified
config['evaluate']['recon_mesh_dir'] = config['experiment']['recon_meshes_dir']
config['evaluate']['gt_mesh_dir'] = '...' # To be specified
config['evaluate']['num_surface_samples'] = 100000
config['evaluate']['f_score_tau'] = 0.01
