import os
import sys
import importlib
from datetime import datetime
import plyfile
import numpy as np
import jittor as jt
from skimage.measure import marching_cubes
from network import Network, CodeCloud
from dataset import ShapeNet


def optimize_latent_codes(network, dataset, shape_idx, config):
    network.code_cloud = CodeCloud(config['network']['code_cloud'], 1)
    network.code_cloud.train()
    network.decoder.eval()
    optimizer = jt.nn.Adam(network.code_cloud.parameters(), config['reconstruct']['init_lr'])
    lr_scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=config['reconstruct']['lr_decay_step'], gamma=config['reconstruct']['lr_decay_rate'])
    for iter_idx in range(config['reconstruct']['num_iteration']):
        query_points, gt_sd, _ = dataset[shape_idx]
        query_points = jt.array(query_points).unsqueeze(0)
        gt_sd = jt.array(gt_sd).unsqueeze(0)
        indices = jt.zeros(1, dtype='int32')
        pred_sd = network(indices, query_points)
        loss_dict = network.loss(gt_sd)
        optimizer.zero_grad()
        optimizer.step(loss_dict['total_loss'])
        lr_scheduler.step()
        print('Optimizing latent code, progress: %d/%d...' % (iter_idx, config['reconstruct']['num_iteration']), end='\r')
    print('Optimizing latent code, progress: %d/%d. Done.' % (config['reconstruct']['num_iteration'], config['reconstruct']['num_iteration']))


def latent_codes_to_mesh(network, config):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Reconstructing mesh(marching cubes resolution=%d).'%config['reconstruct']['marching_cubes_resolution'])

    N = config['reconstruct']['marching_cubes_resolution']
    voxel_size = 1.1 / (N - 1)
    voxel_origin = [-0.55, -0.55, -0.55]

    overall_index = jt.arange(0, N**3, 1, dtype='int32')
    query_points = jt.zeros((N**3, 3))
    query_points[:, 2] = overall_index % N
    query_points[:, 1] = (overall_index / N).long() % N
    query_points[:, 0] = ((overall_index / N) / N).long() % N

    query_points[:, 0] = (query_points[:, 0] * voxel_size) + voxel_origin[2]
    query_points[:, 1] = (query_points[:, 1] * voxel_size) + voxel_origin[1]
    query_points[:, 2] = (query_points[:, 2] * voxel_size) + voxel_origin[0]
    query_points_sdf = np.zeros(N**3)

    network.eval()
    query_points.requires_grad = False
    batch_indices = jt.zeros(1, dtype='int32')
    batch_size = 1024
    num_query_points = N ** 3
    query_point_index = 0
    while query_point_index < num_query_points:
        batch_query_points = query_points[query_point_index : min(query_point_index+batch_size, num_query_points)].unsqueeze(0)
        pred_sd = network(batch_indices, batch_query_points).squeeze(0)
        query_points_sdf[query_point_index : min(query_point_index+batch_size, num_query_points)] = pred_sd.numpy()
        query_point_index += batch_size

    sdf_values = query_points_sdf.reshape(N, N, N)
    vertices, faces, _, _ = marching_cubes(sdf_values, level=0.0, spacing=[voxel_size]*3)
    vertices[:, 0] += voxel_origin[0]
    vertices[:, 1] += voxel_origin[1]
    vertices[:, 2] += voxel_origin[2]
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Reconstruction done.')
    return vertices, faces


def reconstruct(config):
    dataset = ShapeNet(config['recon_dataset'])
    network = Network(config['network'], 1)
    network.decoder.load_parameters(jt.load(config['reconstruct']['ckpt_decoder']))

    for shape_idx in range(len(dataset)):
        print('****** %s ******\ntime: %s\ndata index: %d/%d' % (config['experiment']['name'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'), shape_idx, len(dataset)))
        optimize_latent_codes(network, dataset, shape_idx, config)

        data_id = dataset.split[shape_idx]
        save_path = os.path.join(config['experiment']['recon_latent_codes_dir'], data_id+'.pkl')
        save_path_dir = os.path.dirname(save_path)
        os.makedirs(save_path_dir, exist_ok=True)
        network.code_cloud.save(save_path)
        print('Optimized latent code is saved to ' + save_path)

        vertices, faces = latent_codes_to_mesh(network, config)
        vertices_tuple = np.zeros((vertices.shape[0],), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        for i in range(vertices.shape[0]):
            vertices_tuple[i] = tuple(vertices[i, :])
        faces_tuple = []
        for i in range(faces.shape[0]):
            faces_tuple.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_tuple, dtype=[('vertex_indices', 'i4', (3,))])
        ply_data = plyfile.PlyData([plyfile.PlyElement.describe(vertices_tuple, 'vertex'), plyfile.PlyElement.describe(faces_tuple, 'face')])
        save_path = os.path.join(config['experiment']['recon_meshes_dir'], data_id+'.ply')
        save_path_dir = os.path.dirname(save_path)
        os.makedirs(save_path_dir, exist_ok=True)
        ply_data.write(save_path)
        print('Reconstructed mesh is saved to ' + save_path)


if __name__ == '__main__':
    config = importlib.import_module(sys.argv[1]).config
    jt.flags.use_cuda = 1
    os.makedirs(config['experiment']['recon_latent_codes_dir'])
    os.makedirs(config['experiment']['recon_meshes_dir'])
    reconstruct(config)
