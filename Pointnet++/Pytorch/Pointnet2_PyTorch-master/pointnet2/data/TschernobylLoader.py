from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import sys
import subprocess
import shlex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name, additional_data=False):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:]
    if additional_data:
        intens = f['intens'][:]/255.0  # normalize intensity values to [0,1]
        # print(intens.max())
        # print(intens.min())
        shape = intens.shape
        # cnt = 0
        # copy = intens[0, 0]
        # with np.nditer(intens, op_flags=['readwrite']) as it:
        #     for x in it:
        #         if cnt % shape[1] == 0:
        #             copy = x
        #         x[...] = copy
        #         cnt += 1
        intens_reshp = np.reshape(intens, (-1, shape[1], 1))
        normal = f['normal'][:]

        f_1 = f['feature'][:, :, 0]
        f_2 = f['feature'][:, :, 1]
        f_3 = f['feature'][:, :, 2]
        f_4 = f['feature'][:, :, 3]
        f_5 = f['feature'][:, :, 4]
        # feature_old = f['feature'][:, :, :]  # define here which features should be used
        feature = np.stack([f_1, f_2, f_3, f_4, f_5], axis=2)
        # feature = np.reshape(feature, (-1, shape[1], 1))

        # ############# SB #############
        # t, i = np.concatenate([data], -1), label  # 1_GEOM
        # t, i = np.concatenate([data, normal], -1), label  # 2_GEOM_normals
        # t, i = np.concatenate([data, normal, intens_reshp], -1), label  # 3_GEOM_normals_int/EW
        t, i = np.concatenate([data, normal, intens_reshp, feature], -1), label  # 4_GEOM_normals_int/EW_top5MS

        return t, i
    return data, label


class TschernobylCls(data.Dataset):
    def __init__(self, num_points, transforms=None, train=True, additional_data=False):
        super().__init__()
        self.additional_data = additional_data      # TODO
        self.transforms = transforms

        self.folder = "npbw/Gabriel" # "npbw/SB"; "tschernobyl/all_tree_SB_normalizedFeat"; # "tschernobyl/all_tree_normalized1024_0.7_augmented_r_-1_idoff_50000"
        self.data_dir = os.path.join(BASE_DIR, self.folder)

        if os.path.exists(self.data_dir) is False:
            print("Warning. File doesn't exist.\nAbort")
            sys.exit(0)

        self.train, self.num_points = train, num_points
        if self.train:
            self.files = _get_data_files(os.path.join(self.data_dir, "train_files.txt"))
        else:
            self.files = _get_data_files(os.path.join(self.data_dir, "test_files.txt"))

        point_list, label_list, normals_list, intens_list = [], [], [], []
        for f in self.files:
                points, labels = _load_data_file(os.path.join(BASE_DIR, f), additional_data=self.additional_data)
                point_list.append(points)
                label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

        # self.randomize() TODO roessl
        self.actual_number_of_points = num_points  # TODO roessl

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.actual_number_of_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        # if self.additional_data:
        #     normals = torch.from_numpy(self.normals[idx]).type(torch.LongTensor)
        #     intens = torch.from_numpy(self.intens[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        # if self.additional_data:
        #     return current_points, label, normals, intens
        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = pts
        self.actual_number_of_points = pts

    def randomize(self):
        self.actual_number_of_points = min(
            max(np.random.randint(self.num_points * 0.8, self.num_points * 1.2), 1),
            self.points.shape[1],
        )


'''
if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )
    dset = TschernobylCls(16, "./", train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
'''