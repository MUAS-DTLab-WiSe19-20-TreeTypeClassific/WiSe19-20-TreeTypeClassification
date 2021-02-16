from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import os
import argparse

from pointnet2.models import Pointnet2ClsMSG as Pointnet
from pointnet2.models.pointnet2_msg_cls import model_fn_decorator
from pointnet2.data.TschernobylLoader import TschernobylCls
import pointnet2.data.data_utils as d_utils

from etw_pytorch_utils.utility_cm import make_res_folder, save_hyper  # roessl
from time import time


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")  # 16
    parser.add_argument(
        "-num_points", type=int, default=1024, help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay", type=float, default=1e-4, help="L2 regularization coeff"  # 1e-5
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")  # 1e-2
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"  # 0.7
    )
    parser.add_argument(
        "-decay_step", type=float, default=8e3, help="Learning rate decay step"  # 2e5
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"  # 0.5
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"   # 0.5
    )
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-epochs", type=int, default=100, help="Number of epochs to train for"  #200
    )
    parser.add_argument(
        "-run_name",
        type=str,
        default="cls_run_1",
        help="Name for run in tensorboard_logger",
    )
    parser.add_argument("--visdom-port", type=int, default=None)  # 8097
    parser.add_argument("--visdom", action="store_false")  # store_true

    return parser.parse_args()


lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    t_start = time()
    args = parse_args()

    transforms_train = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale(),
            d_utils.PointcloudRotate(),
            d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
            d_utils.PointcloudRandomInputDropout(),
        ]
    )

    a_data = True   # use additional data like normals, feature, intensity etc. TODO
    save_valid = True   # parameter to save each epochs validation result; (Shuffle of the test data should be False)
    test_set = TschernobylCls(args.num_points, transforms=None, train=False, additional_data=a_data)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        # shuffle=True,  # TODO roessl-> stop shuffling of test data, in order to compare the label with the prediction
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    train_set = TschernobylCls(args.num_points, transforms=transforms_train, additional_data=a_data)  # None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    # TODO input_channels & classes
    model = Pointnet(input_channels=9, num_classes=4, use_xyz=True)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_lbmd = lambda it: max(
        args.lr_decay ** (int(it * args.batch_size / args.decay_step)),
        lr_clip / args.lr,
    )
    bn_lbmd = lambda it: max(
        args.bn_momentum
        * args.bnm_decay ** (int(it * args.batch_size / args.decay_step)),
        bnm_clip,
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status
    # learning rate
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=it)
    # Batch norm momentum
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bn_lbmd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    if args.visdom:
        viz = pt_utils.VisdomViz(port=args.visdom_port)
    else:
        viz = pt_utils.CmdLineViz()

    viz.text(pprint.pformat(vars(args)))

    if not osp.isdir("checkpoints"):
        os.makedirs("checkpoints")

    trainer = pt_utils.Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name="checkpoints/pointnet2_cls",
        best_name="checkpoints/pointnet2_cls_best",
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        viz=viz,
    )

    # === Roessl ===
    res_folder = make_res_folder(path="Default")
    hyper_param = list(map(list, vars(args).items()))
    hyper_param.append(["lr_clip", lr_clip])
    hyper_param.append(["bnm_clip", bnm_clip])
    hyper_param.append(["Dataset", train_set.folder])
    save_hyper(res_folder, hyper_param)
    trainer.train(
        it, start_epoch, args.epochs, train_loader, test_loader, best_loss=best_loss, folder=res_folder,
        save_validation=save_valid
    )

    if start_epoch == args.epochs:
        _ = trainer.eval_epoch(test_loader)
    t_end = time()
    t_dur = t_end - t_start
    print("Training time: %d [h], %d [min], %.2f [sec]" % (t_dur // 3600, t_dur % 3600 // 60, t_dur % 3600 % 60))
