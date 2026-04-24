import sys
sys.path.append('/PATH/TO/Wildfire-SSR')
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from datasets.make_data_loader import make_data_loader
from utils_func.metrics import Evaluator
from utils_func.WarmupLR import WarmupLR
import utils_func.lovasz_loss as L
import utils_func.weighted_ce_loss as WCE
from utils_func.ssr_helpers import build_model_tag, build_write_logfile, get_model


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.write_logfile, stamp = build_write_logfile(args.model_tag, args.resume)

        args.max_epochs = args.max_epochs - args.start_epoch

        args.type = 'train'
        self.train_data_loader = make_data_loader(args)

        val_args = argparse.Namespace(**vars(args))
        val_args.type = 'val'
        val_args.dataset_path = args.test_dataset_path
        val_args.data_list_path = args.test_data_list_path
        val_args.data_name_list = args.test_data_name_list
        val_args.max_epochs = 1
        val_args.batch_size = 1
        val_args.shuffle = False
        val_args.crop_size = 1024
        self.val_data_loader = make_data_loader(val_args)

        args.num_classes = 5
        args.num_loc_classes = 2

        self.evaluator_loc = Evaluator(num_class=args.num_loc_classes + 1 if args.num_loc_classes == 1 else args.num_loc_classes)
        self.evaluator_clf = Evaluator(num_class=args.num_classes)

        self.deep_model = get_model(args, freeze_backbone=args.freeze_backbone)
        self.deep_model = self.deep_model.cuda()
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            f'{args.model_tag}_{stamp}')
        self.lr = args.learning_rate

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model_dict = {k: v for k, v in checkpoint.items() if k in self.deep_model.state_dict()}
            self.deep_model.load_state_dict(model_dict, strict=False)
            self.write_logfile(f"Resumed from checkpoint: {args.resume}")

        self.write_logfile("Setting up optimizer with differential learning rates...")

        if self.args.freeze_backbone:
            print("Freezing all encoder.backbone parameters.")
            for name, param in self.deep_model.named_parameters():
                if name.startswith('encoder.backbone'):
                    param.requires_grad = False

        backbone_params = []
        new_params = []

        for name, param in self.deep_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('encoder.backbone'):
                backbone_params.append(param)
            else:
                new_params.append(param)

        self.write_logfile(f"Found {len(backbone_params)} trainable backbone params and {len(new_params)} new params.")
        self.write_logfile(f"Batch size: {args.batch_size}, Accumulation steps: {args.accumulation_steps}")
        self.write_logfile(f"Backbone params LR: {args.learning_rate / 10}, New params LR: {args.learning_rate}")
        self.write_logfile(f"Weight decay: {args.weight_decay}")
        self.write_logfile(f"CE Loss weights - loc: {args.ce_loss_loc_weight}, clf: {args.ce_loss_clf_weight}")
        self.write_logfile(f"Lovasz Loss weights - loc: {args.lovasz_loss_loc_weight}, clf: {args.lovasz_loss_clf_weight}")
        self.write_logfile(f"Classification Class weights - clf: {args.clf_cls_weights}")
        self.write_logfile(f"Model tag: {args.model_tag}")

        param_groups = [
            {'params': backbone_params, 'lr': args.learning_rate / 10, 'weight_decay': args.weight_decay},
            {'params': new_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay}
        ]

        self.optim = optim.AdamW(param_groups)
        self.accumulation_steps = args.accumulation_steps

        if args.resume is not None:
            self.lr_scheduler = WarmupLR(self.optim, warmup_steps=10)

    def training(self):
        if self.args.resume is not None:
            self.write_logfile(f'---------Resume from {self.args.resume[:-4]}---------')
        else:
            self.write_logfile('---------Start training---------')

        best_harmonic_mean_f1 = 0.0
        best_epoch = 0
        torch.cuda.empty_cache()

        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        data_num = len(self.args.train_data_name_list)

        for _ in tqdm(range(elem_num), desc="Training Progress"):
            itera, data = train_enumerator.__next__()
            pre_imgs, labels_loc, labels_clf, texts_data, _ = data

            pre_imgs = pre_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()

            output_loc, output_clf = self.deep_model(pre_imgs, texts_data)

            ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
            lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
            loss_loc = (self.args.ce_loss_loc_weight * ce_loss_loc + self.args.lovasz_loss_loc_weight * lovasz_loss_loc).mean()

            clf_weights = torch.tensor(self.args.clf_cls_weights).cuda()
            ce_loss_clf = WCE.distance_weighted_ce_loss(output_clf, labels_clf, class_weights=clf_weights, ignore_index=255, alpha=0.5)
            lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)
            loss_clf = (self.args.ce_loss_clf_weight * ce_loss_clf + self.args.lovasz_loss_clf_weight * lovasz_loss_clf).mean()

            final_loss = (loss_loc + loss_clf) / self.accumulation_steps
            final_loss.backward()

            if ((itera + 1) % self.accumulation_steps) == 0:
                self.optim.step()
                if self.args.resume is not None:
                    self.lr_scheduler.step()
                self.optim.zero_grad()

            if (itera + 1) % 10 == 0:
                self.write_logfile(f'iter is {itera + 1}, localization loss is {loss_loc.item():.4f}, classification loss is {loss_clf.item():.4f}')

            if (itera * self.args.batch_size) % data_num >= (data_num - self.args.batch_size):
                epoch = (itera * self.args.batch_size // data_num) + 1 + self.args.start_epoch
                torch.save(self.deep_model.state_dict(), os.path.join(self.model_save_path, f'latest_model.pth'))

                self.deep_model.eval()
                self.write_logfile(f'epoch is {epoch}, localization loss is {loss_loc.item():.4f}, classification loss is {loss_clf.item():.4f}')
                loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score = self.validation()

                if harmonic_mean_f1 > best_harmonic_mean_f1:
                    best_harmonic_mean_f1 = harmonic_mean_f1
                    best_epoch = epoch
                    best_scores = [loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score]
                    torch.save(self.deep_model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))

                self.deep_model.train()

        self.write_logfile(f'Best epoch is {best_epoch}')
        self.write_logfile(f'Best scores: locF1 is {best_scores[0]}, clfF1 is {best_scores[1]}, '
                           f'oaF1 is {best_scores[2]}, sub class F1 score is {best_scores[3]}')

    def validation(self):
        print('---------Starting Evaluation-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        torch.cuda.empty_cache()

        with torch.no_grad():
            for data in tqdm(self.val_data_loader, desc="Validation"):
                pre_imgs, labels_loc, labels_clf, texts_data, _ = data
                pre_imgs, labels_loc, labels_clf = pre_imgs.cuda(), labels_loc.cuda().long(), labels_clf.cuda().long()

                output_loc, output_clf = self.deep_model(pre_imgs, texts_data)

                preds_loc = torch.argmax(output_loc, dim=1).cpu().numpy()
                labels_loc = labels_loc.cpu().numpy()
                self.evaluator_loc.add_batch(labels_loc, preds_loc)

                preds_clf = torch.argmax(output_clf, dim=1).cpu().numpy()
                labels_clf = labels_clf.cpu().numpy()

                valid_mask = labels_clf != 255
                self.evaluator_clf.add_batch(labels_clf[valid_mask], preds_clf[valid_mask])

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / (damage_f1_score + 1e-8))
        oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

        self.write_logfile(f'locF1 is {loc_f1_score}, clfF1 is {harmonic_mean_f1}, oaF1 is {oaf1}, '
                           f'sub class F1 score is {damage_f1_score}')
        return loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score


def main():
    parser = argparse.ArgumentParser(description="Training on TextRSDataset with VisionLanguageSSR")
    parser.add_argument('--dataset', type=str, default='TextRSDataset')
    parser.add_argument('--train_dataset_path', type=str, default='../data')
    parser.add_argument('--train_data_list_path', type=str, default='../data/splits/Train_dataset.txt')
    parser.add_argument('--test_dataset_path', type=str, default='../data')
    parser.add_argument('--test_data_list_path', type=str, default='../data/splits/Valid_dataset.txt')
    parser.add_argument('--model_param_path', type=str, default='./saved_models')
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--shuffle', type=lambda x: x.lower() == 'true', default=True)

    parser.add_argument('--model_size', type=str, default='Large',
                        choices=['Small', 'Base', 'Large', '7B'])
    parser.add_argument('--backbone_weights', type=str,
                        default='models/dinov3/dinov3/pretrained_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
                        help='Path to the DINOv3 backbone pretrained weights')
    parser.add_argument('--geo_embed_type', type=str, default='RoPE',
                        choices=['RoPE', 'RFF', 'none'],
                        help="Geolocation embedding type for the decoder")
    parser.add_argument('--ablated_vision', action='store_true',
                        help="If set, replace DINOv3 features with zero tensors (ablated vision)")
    parser.add_argument('--freeze_backbone', type=lambda x: x.lower() == 'true', default=True,
                        help='Freeze the DINOv3 backbone parameters')

    parser.add_argument('--cuda', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--accumulation_steps', type=int, default=2)

    parser.add_argument('--ce_loss_loc_weight', type=float, default=1.0)
    parser.add_argument('--ce_loss_clf_weight', type=float, default=1.0)
    parser.add_argument('--lovasz_loss_loc_weight', type=float, default=0.5)
    parser.add_argument('--lovasz_loss_clf_weight', type=float, default=0.75)
    parser.add_argument('--clf_cls_weights', type=str, default="1.,1.,1.,1.,1.")
    args = parser.parse_args()
    args.clf_cls_weights = [float(w) for w in args.clf_cls_weights.split(',')]

    args.model_tag = build_model_tag(args)

    args.dataset_path = args.train_dataset_path
    args.data_list_path = args.train_data_list_path

    with open(args.train_data_list_path, "r") as f:
        args.train_data_name_list = [name.strip() for name in f]
        print(f"Number of training samples found: {len(args.train_data_name_list)}")
    with open(args.test_data_list_path, "r") as f:
        args.test_data_name_list = [name.strip() for name in f]
        print(f"Number of validation samples found: {len(args.test_data_name_list)}")

    args.data_name_list = args.train_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
