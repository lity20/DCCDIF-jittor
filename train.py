import os
import sys
import json
import importlib
from datetime import datetime
import jittor as jt
from network import Network
from dataset import ShapeNet


def prepare(config):
    jt.flags.use_cuda = 1
    os.makedirs(config['experiment']['dir'])
    with open(os.path.join(config['experiment']['dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    src_dir = os.path.join(config['experiment']['dir'], 'src')
    os.makedirs(src_dir)
    os.system('cp *.py ' + src_dir)
    os.makedirs(config['experiment']['ckpt_save_dir'])


def train(config):
    prepare(config)
    train_dataset = ShapeNet(config['train_dataset'])
    network = Network(config['network'], len(train_dataset.split))
    decoder_optimizer = jt.nn.Adam(network.decoder.parameters(), config['train']['decoder_init_lr'])
    decoder_lr_scheduler = jt.lr_scheduler.StepLR(decoder_optimizer, step_size=config['train']['decoder_lr_decay_step'], gamma=config['train']['decoder_lr_decay_rate'])
    latent_codes_optimizer = jt.nn.Adam(network.code_cloud.parameters(), config['train']['latent_codes_init_lr'])
    latent_codes_lr_scheduler = jt.lr_scheduler.StepLR(latent_codes_optimizer, step_size=config['train']['latent_codes_lr_decay_step'], gamma=config['train']['latent_codes_lr_decay_rate'])

    network.train()
    for epoch in range(config['train']['num_epoch']):
        print('****** %s ******\ntime: %s\nepoch: %d/%d' % (config['experiment']['name'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, config['train']['num_epoch']))
        for batch_idx, (query_points, gt_sd, indices) in enumerate(train_dataset):
            pred_sd = network(indices, query_points)
            loss_dict = network.loss(gt_sd)
            decoder_optimizer.zero_grad()
            decoder_optimizer.step(loss_dict['total_loss'])
            latent_codes_optimizer.zero_grad()
            latent_codes_optimizer.step(loss_dict['total_loss'])
            print('Current epoch progress: %d/%d...' % (batch_idx, len(train_dataset)), end='\r')
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Current epoch progress: %d/%d. Done.' % (len(train_dataset), len(train_dataset)))
        decoder_lr_scheduler.step()
        latent_codes_lr_scheduler.step()

        for k, v in loss_dict.items():
            print(k, ':', v.numpy()[0])
        print('decoder lr :', decoder_optimizer.lr)
        print('latent codes lr :', latent_codes_optimizer.lr)

    network.decoder.save(os.path.join(config['experiment']['ckpt_save_dir'], 'decoder.pkl'))
    network.code_cloud.save(os.path.join(config['experiment']['ckpt_save_dir'], 'latent_codes.pkl'))
    print('Trained models are saved to', config['experiment']['ckpt_save_dir'])


if __name__ == '__main__':
    config = importlib.import_module(sys.argv[1]).config
    train(config)
