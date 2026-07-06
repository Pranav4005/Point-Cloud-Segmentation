import open3d as o3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
from nearest_neighbors import nearest_neighbors


class ConfigSemanticKITTI:
    k_n = 16
    num_layers = 4
    num_points = 4096 * 4
    num_classes = 8
    sub_grid_size = 0.06

    batch_size = 2
    val_batch_size = 2
    train_steps = 30
    val_steps = 10

    sub_sampling_ratio = [4, 4, 4, 4]
    d_out = [16, 64, 128, 256]
    num_sub_points = [
        num_points // 4,
        num_points // 16,
        num_points // 64,
        num_points // 256
    ]

    noise_init = 3.5
    max_epoch = 28
    learning_rate = 1e-4
    lr_decays = {i: 0.95 for i in range(0, 500)}

    train_sum_dir = '/content/drive/MyDrive/RandLA_DALES_backup/train_log'
    saving = True
    saving_path = '/content/drive/MyDrive/RandLA_DALES_backup/results'


class DataProcessing:

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        support_pts: B*N1*3
        query_pts: B*N2*3
        """
        neighbor_idx = nearest_neighbors.knn_batch(
            support_pts, query_pts, k, omp=True
        )
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)

        dup = np.random.choice(num_in, num_out - num_in)

        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)

        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)

        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]

        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(len(data_list))
        np.random.shuffle(indices)
        data_list = np.array(data_list)[indices]
        return data_list.tolist()

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(
                points,
                sampleDl=grid_size,
                verbose=verbose
            )

        elif labels is None:
            return cpp_subsampling.compute(
                points,
                features=features,
                sampleDl=grid_size,
                verbose=verbose
            )

        elif features is None:
            return cpp_subsampling.compute(
                points,
                classes=labels,
                sampleDl=grid_size,
                verbose=verbose
            )

        else:
            return cpp_subsampling.compute(
                points,
                features=features,
                classes=labels,
                sampleDl=grid_size,
                verbose=verbose
            )

    @staticmethod
    def IoU_from_confusions(confusions):

        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(dataset_name):

        num_per_class = []

        if dataset_name == 'DALES':
            num_per_class = np.array([
                1000000,
                800000,
                200000,
                100000,
                50000,
                150000,
                100000,
                300000
            ], dtype=np.int32)

        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)

        return np.expand_dims(ce_label_weight, axis=0)


class Plot:

    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]

        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

        random.seed(seed)
        random.shuffle(colors)

        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])

        if pc_xyzrgb.shape[1] == 3:
            o3d.visualization.draw_geometries([pc])
            return 0

        if np.max(pc_xyzrgb[:, 3:6]) > 20:
            pc.colors = o3d.utility.Vector3dVector(
                pc_xyzrgb[:, 3:6] / 255.
            )
        else:
            pc.colors = o3d.utility.Vector3dVector(
                pc_xyzrgb[:, 3:6]
            )

        o3d.visualization.draw_geometries([pc])

        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):

        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(
                len(np.unique(pc_sem_ins)) + 1,
                seed=2
            )

        sem_ins_labels = np.unique(pc_sem_ins)
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))

        for idx, semins in enumerate(sem_ins_labels):

            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]

            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[idx]

            Y_colors[valid_ind] = tp

        Y_semins = np.concatenate(
            [pc_xyz[:, 0:3], Y_colors],
            axis=-1
        )

        Plot.draw_pc(Y_semins)

        return Y_semins