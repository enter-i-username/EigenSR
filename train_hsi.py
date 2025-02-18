import utils
import shutil
from dataset import transforms
from dataset import get_dataset
from degradation import ImgDownscale
from model import grayscale_model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import model.inference as inference
import trainer.trainer as trainer
from loss.loss import LossLog
from metrics.metrics import MetricLog, get_metric
import numpy as np
from tqdm import tqdm
import os
from importlib.machinery import SourceFileLoader



if __name__ == '__main__':
    # load config file
    config_fn = 'train_config.py'
    config = SourceFileLoader('config', config_fn).load_module()

    # copy train_config.py to exp_folder
    config.exp_folder = os.path.join(
        config.exp_folder,
        'x{scale}_{train_dataset}_{test_dataset}_{current_time}'
    ).format(
        scale=config.scale,
        train_dataset=config.train_dataset,
        test_dataset=config.test_dataset,
        current_time=utils.get_current_time_str()
    )

    dst = os.path.join(config.exp_folder, config_fn)
    utils.make_dirs_if_not_exist(dst)
    shutil.copy2(config_fn, dst)

    # init
    restore = grayscale_model.GrayscaleSR(
        scale=config.scale,
        hidden_channels=config.hidden_channels,
        need_lora=config.need_lora,
        lora_r=config.lora_r
    ).load_frozen_body(config.vit_body_fn)

    model_parallel = False
    if config.device.lower().find('cuda') >= 0 and \
            len(config.device_ids) > 1:
        restore = nn.DataParallel(restore, device_ids=config.device_ids)
        model_parallel = True

    train_set = get_dataset(config.dataset_root, config.train_dataset, 'train', config.dataset_cache,
                            return_keys=['hsi', 'svd'])
    train_set = transforms.Transform(
        dataset=train_set,
        transform=transforms.Sequence(
            (transforms.FromDataDictKeys(keys=('hsi', 'svd')),
             transforms.NumpyToTorch(keys=('hsi', 'svd')),
             transforms.TensorChannelFirst(keys=('hsi', )),

             transforms.TensorRandomCrop(size=config.train_crop_size, keys=('hsi', )),
             transforms.TensorRandomRotate(keys=('hsi', )),
             transforms.TensorRandomFlip(p=config.train_random_flip_p, keys=('hsi', )),

             transforms.GetSeqDegHSI(
                 degradation_lst=(ImgDownscale(scale_factor=1 / config.scale), ),
                 hsi_key='hsi', deg_hsi_key='deg_hsi'
             ),
             transforms.GetRandomSVDEigenimage(
                 thr=config.train_svd_threshold,
                 hsi_key='hsi', deg_hsi_key='deg_hsi', svd_key='svd',
                 eigen_key='eigen', deg_eigen_key='deg_eigen'
             ),
             transforms.NormPair(src_key='deg_eigen', tgt_key='eigen', Norm=utils.Norm),

             transforms.ReturnKeys(keys=('deg_eigen', 'eigen'))
             )
        )
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.train_workers,
    )

    if config.test_dataset is not None:
        test_set = get_dataset(config.dataset_root, config.test_dataset, 'test', config.dataset_cache)
        test_set = transforms.Transform(
            dataset=test_set,
            transform=transforms.Sequence(
                (transforms.FromDataDictKeys(keys=('hsi', )),
                 transforms.NumpyToTorch(keys=('hsi', )),
                 transforms.TensorChannelFirst(keys=('hsi', )),

                 transforms.GetSeqDegHSI(
                     degradation_lst=(ImgDownscale(scale_factor=1 / config.scale), ),
                     hsi_key='hsi', deg_hsi_key='deg_hsi'
                 ),

                 transforms.ReturnKeys(keys=('deg_hsi', 'hsi'))
                 )
            )
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.test_workers,
        )

    trainable_params = filter(lambda p: p.requires_grad, restore.parameters())
    optimizer = torch.optim.Adam(params=trainable_params, lr=config.lr)
    linear_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _epoch: max(0.0, 1.0 - _epoch / config.num_epochs)
    )

    loss_log = LossLog()
    metric_log = MetricLog(
        **{_k: get_metric(_k) for _k in config.metric_keys}
    )


    def get_epoch_fn(exp_folder, fn, epoch):
        fn = os.path.join(exp_folder, fn).format(epoch=epoch)
        utils.make_dirs_if_not_exist(fn)
        return fn


    print(f'\n\nExperment started at {config.exp_folder}\n\n')
    for epoch in tqdm(range(1, 1 + config.num_epochs)):

        trainer.train_one_epoch(
            model=restore,
            train_dataloader=train_loader,
            loss_log=loss_log,
            optimizer=optimizer,
            device=config.device,
            lr_scheduler=linear_scheduler if config.lr_linear_decay else None
        )

        if epoch % config.trainable_save_every == 0:
            _fn = get_epoch_fn(config.exp_folder, config.trainable_params_fn, epoch)
            if model_parallel:
                restore.module.save_trainable(_fn)
            else:
                restore.save_trainable(_fn)

        if epoch % config.optimizer_save_every == 0:
            _fn = get_epoch_fn(config.exp_folder, config.optimizer_fn, epoch)
            torch.save(
                {'optimizer_state_dict': optimizer.state_dict()},
                _fn
            )

        if epoch % config.loss_log_save_every == 0:
            _fn = get_epoch_fn(config.exp_folder, config.loss_log_fn, epoch)
            loss_log.write_save(epoch, _fn)

        if config.test_dataset is not None:
            if epoch % config.test_every == 0:
                trainer.test(
                    model=restore,
                    test_dataloader=test_loader,
                    inference_func=lambda _hsi, _model: inference.iterative_sr(
                        # iter params
                        deg_hsi_batch=_hsi,
                        iters=1,
                        beta=1,
                        # hsi_inference params
                        model=_model,
                        scale=config.scale,
                        batch_size=config.inference_batch_size,
                        patch_size=(48, 48),
                        step_size=config.inference_step_size,
                        eigen_dims=config.inference_eigen_dims,
                        transform=False
                    ),
                    metric_log=metric_log,
                    device=config.device
                )

            if epoch % config.metric_log_save_every == 0:
                _fn = get_epoch_fn(config.exp_folder, config.metric_log_fn, epoch)
                metric_log.write_save(
                    epoch, _fn, metric=config.metric_optimal_key, optimal=config.metric_optimal
                )

            if epoch == metric_log.get_best_epoch():
                _fn = get_epoch_fn(config.exp_folder, config.best_trainable_fn, epoch)
                if model_parallel:
                    restore.module.save_trainable(_fn)
                else:
                    restore.save_trainable(_fn)









