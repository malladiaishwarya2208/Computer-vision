# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/MAL/blob/main/LICENSE


import json
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torch

try:
    import argparse
    import os
    import importlib.util
    from models.mal import MAL, MALPseudoLabels
    from torch import nn
    import torch.nn.functional as F
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import loggers as pl_loggers
    from datasets.pl_data_module import WSISDataModule, datapath_configs
except:
    pass

print('\n\n')
print('\t',"torch.cuda.is_available() = ",torch.cuda.is_available(),'\t')
print('\t',"torch.cuda.device_count() = ",torch.cuda.device_count(),'\t')
print('\n\n')


def parse_args():
    parser = argparse.ArgumentParser()

    # load config
    parser.add_argument("--val_only", action='store_true', default=False)
    parser.add_argument("--box_inputs", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help='Weight name to be resumed')
    parser.add_argument('--val_interval', default=1, type=int)

    # Dataset
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers_per_gpu', default=2, type=int)
    parser.add_argument('--dataset_type', default='coco', type=str, choices=datapath_configs.keys())

    # Hyperparameter
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--accum_grad_batches', default=1, type=int)
    parser.add_argument('--min_obj_size', default=2048, type=int)
    parser.add_argument('--max_obj_size', default=1e10, type=int)
    parser.add_argument('--strategy', default='ddp_sharded', type=str)
    parser.add_argument('--num_mp_devices', default=None, type=int)

    parser.add_argument('--optim_type', default='adamw', type=str, choices=['sgd', 'adamw'])
    parser.add_argument('--optim_momentum', default=0.9, type=float)
    # parameters for annealLR + adamW
    parser.add_argument('--lr', default=0.0000015, type=float)
    parser.add_argument('--min_lr_rate', default=0.2, type=float)
    parser.add_argument('--num_wave', default=1, type=float)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--optim_eps', default=1e-8, type=float)
    parser.add_argument('--optim_betas', default="0.9,0.9", type=str)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    

    # parameters for sgd
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--save_every_k_epoch', default=1, type=int)

    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--margin_rate', default="0,1.2", type=str)
    parser.add_argument('--test_margin_rate', default='0.6,0.6', type=str)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--mask_thres', default='0.1', type=str)

    # multi-node and multi-gpus
    parser.add_argument('--gpus', default=None, type=str)
    parser.add_argument("--nnodes", default=1, type=int)

    # network architecture
    parser.add_argument('--arch', default='vit-mae-base/16', type=str)
    parser.add_argument('--frozen_stages', default="-1", type=str)
    parser.add_argument('--mask_head_num_convs', default=4, type=int)
    parser.add_argument('--mask_head_hidden_channel', default=256, type=int)
    parser.add_argument('--mask_head_out_channel', default=256, type=int)
    parser.add_argument('--teacher_momentum', default=0.996, type=float)
    parser.add_argument('--not_adjust_scale', action='store_true', default=False)
    parser.add_argument('--mask_scale_ratio_pre', default=1, type=int)
    parser.add_argument('--mask_scale_ratio', default=2.0, type=float)
    parser.add_argument('--vit_dpr', type=float, default=0)

    # transform
    parser.add_argument('--train_transform', default='train', type=str)
    parser.add_argument('--test_transform', default='test', type=str)

    # loss option
    parser.add_argument('--loss_mil_weight', default=4, type=float)
    parser.add_argument('--loss_crf_weight', default=0.5, type=float)

    # crf option
    parser.add_argument('--crf_zeta', default=0.1, type=float)
    parser.add_argument('--crf_omega', default=2, type=float)
    parser.add_argument('--crf_kernel_size', default=3, type=float)
    parser.add_argument('--crf_num_iter', default=100, type=int)
    parser.add_argument('--loss_crf_step', default=4000, type=int)
    parser.add_argument('--loss_mil_step', default=1000, type=int)
    parser.add_argument('--crf_size_ratio', default=1, type=int)
    parser.add_argument('--crf_value_high_thres', default=0.9, type=float)
    parser.add_argument('--crf_value_low_thres', default=0.1, type=float)

    # inference
    parser.add_argument('--use_mixed_model_test', action='store_true', default=False)
    parser.add_argument('--use_teacher_test', action='store_true', default=False)
    parser.add_argument('--use_flip_test', action='store_true', default=False)
    parser.add_argument('--use_crf_test', action='store_true', default=False)
    parser.add_argument('--not_eval_mask', action='store_true', default=False)
    parser.add_argument('--comp_clustering', action='store_true', default=False)

    # Generating mask pseudo-labels
    parser.add_argument('--label_dump_path', default=None, type=str)



    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()

        # random seed
        seed_everything(args.seed)

        # log
        tb_logger = pl_loggers.TensorBoardLogger(os.path.join("tb_logs", args.dataset_type)) 

        # learning rate
        # optim betas
        args.optim_betas = list(map(float, args.optim_betas.split(",")))
        # margin rate
        args.margin_rate = list(map(float, args.margin_rate.split(",")))
        args.test_margin_rate = list(map(float, args.test_margin_rate.split(",")))
        # mask threshold
        args.mask_thres = list(map(float, args.mask_thres.split(",")))
        if len(args.mask_thres) == 1:
            # this means to repeat the same threshold three times
            # all scale objects are sharing the same threshold
            args.mask_thres = [args.mask_thres[0] for _ in range(3)]
        assert len(args.mask_thres) == 3
        # frozen_stages
        args.frozen_stages = list(map(int, args.frozen_stages.split(",")))
        if len(args.frozen_stages) == 1:
            args.frozen_stages = [0, args.frozen_stages[0]]
        assert len(args.frozen_stages) == 2

        if len(args.margin_rate) == 1:
            args.margin_rate = args.margin_rate[0]
        elif len(args.margin_rate) == 2:
            pass
        else:
            raise NotImplementedError

        if args.gpus is not None:
            args.gpus = list(map(int, args.gpus.split(",")))
        else:
            args.gpus = list(range(torch.cuda.device_count()))

        args.lr = args.lr * len(args.gpus) * args.batch_size
        args.min_lr = args.lr * args.min_lr_rate

        num_workers = len(args.gpus) * args.num_workers_per_gpu

        data_loader = WSISDataModule(num_workers=num_workers, 
                                     load_train=not args.val_only and args.label_dump_path is None,
                                     load_val=True, args=args)
        
        if not args.val_only and args.label_dump_path is None:
            num_iter_per_epoch = len(data_loader.train_dataloader())
        else:
            num_iter_per_epoch = 1

        if args.label_dump_path is not None:
            # Phase 2: Generating pseudo-labels
            model = MALPseudoLabels(args=args)
            trainer = Trainer(gpus=args.gpus, strategy=args.strategy, devices=args.num_mp_devices, accelerator='gpu',
                            precision=16, check_val_every_n_epoch=args.val_interval,
                            logger=tb_logger, resume_from_checkpoint=args.resume)#, fast_dev_run=3)
            trainer.validate(model, ckpt_path=args.resume, dataloaders=data_loader.val_dataloader())
        else:
            # Phase 1: Training and testing MAL
            if args.box_inputs is not None:
                model = MALPseudoLabels(args=args, num_iter_per_epoch=num_iter_per_epoch)
            else:
                model = MAL(args=args, num_iter_per_epoch=num_iter_per_epoch)

            checkpoint_callback = ModelCheckpoint(
                                    dirpath=os.path.join("work_dirs", args.dataset_type),
                                    filename='{epoch}' + \
                                             '-arch={}'.format(args.arch.replace("/", "-")) +\
                                             '-not_adjust_scale={}-mask_scale_ratio_pre={}'.format(args.not_adjust_scale, args.mask_scale_ratio_pre),
                                            save_top_k=-1, every_n_epochs=args.save_every_k_epoch, save_last=True)

            trainer = Trainer(gpus=args.gpus, num_nodes=args.nnodes, strategy=args.strategy, devices=args.num_mp_devices,
                            callbacks=[checkpoint_callback], accelerator='gpu', max_epochs=args.max_epochs,
                            precision=16, check_val_every_n_epoch=args.val_interval, 
                            logger=tb_logger, resume_from_checkpoint=args.resume,
                            accumulate_grad_batches=args.accum_grad_batches)

            if not args.val_only:
                trainer.fit(model, data_loader)
            else:
                trainer.validate(model, ckpt_path=args.resume, dataloaders=data_loader.val_dataloader())
    except:

        parser = argparse.ArgumentParser(description='Process COCO dataset for panoptic segmentation.')
        parser.add_argument('--dataset',required=True , type=str, choices=['train2017', 'val2017'], help='Specify the dataset to process: train2017, val2017')
        args = parser.parse_args()
        
        
        
        dataDir = 'data\\coco'  
        annFile = os.path.join(dataDir, 'annotations', f'instances_{args.dataset}.json')
        coco = COCO(annFile)
        
        
        processed_dir = os.path.join('output_images', args.dataset)
        os.makedirs(processed_dir, exist_ok=True)
        
        
        data_store = {
            "images": [],
            "annotations": []
        }


        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(dataDir, args.dataset, img_info['file_name'])
            print("Attempting to open image at:", img_path)  
            try:
                image = Image.open(img_path)
            except FileNotFoundError:
                print(f"File not found: {img_path}")
                continue  
        
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
        
            
            panoptic = np.zeros((img_info['height'], img_info['width']), dtype=np.uint32)
            for ann in anns:
                mask = coco.annToMask(ann) * ann['category_id']
                panoptic = np.maximum(panoptic, mask)
        
            
            panoptic_image = Image.fromarray(panoptic.astype(np.uint8), 'L')
        
            
            processed_img_path = os.path.join(processed_dir, img_info['file_name'])
            panoptic_image.save(processed_img_path, 'PNG')
        
            
            data_store['images'].append({
                "id": img_id,
                "path": processed_img_path
            })
            data_store['annotations'].append({
                "image_id": img_id,
                "segmentation_path": processed_img_path,
                "category_ids": [ann['category_id'] for ann in anns]
            })
        
        json_file_path = os.path.join(processed_dir, 'data_store.json')
        with open(json_file_path, 'w') as f:
            json.dump(data_store, f,indent = 2)
        
        print(f"Processing complete! Data stored in {json_file_path}")