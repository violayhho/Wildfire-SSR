import sys
sys.path.append('/PATH/TO/Wildfire-SSR')

import argparse
import os

import numpy as np
import imageio.v2 as imageio
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

import torch

from datasets.make_data_loader import make_data_loader
from utils_func.metrics import Evaluator
from utils_func.ssr_helpers import build_model_tag, get_model


class Inferencer(object):
    def __init__(self, args):
        args.max_epochs = 1
        args.type = 'test'
        self.args = args

        self.evaluator_loc = Evaluator(num_class=args.num_loc_classes + 1 if args.num_loc_classes == 1 else args.num_loc_classes)
        self.evaluator_clf = Evaluator(num_class=args.num_classes)

        self.deep_model = get_model(args)
        self.deep_model = self.deep_model.cuda()

        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> No checkpoint found at '{args.resume}'. Cannot run inference.")

        print(f"Loading checkpoint from '{args.resume}'")
        checkpoint = torch.load(args.resume)

        state_dict = checkpoint.get('state_dict', checkpoint)

        model_dict = {k: v for k, v in state_dict.items() if k in self.deep_model.state_dict()}
        self.deep_model.load_state_dict(model_dict, strict=False)
        print(f"Successfully loaded checkpoint.")

        self.deep_model.eval()

        if self.args.save_images:
            model_folder_name = os.path.basename(os.path.dirname(args.resume))
            self.clf_map_saved_path = os.path.join(args.result_saved_path, args.dataset, model_folder_name, 'classification_maps')
            os.makedirs(self.clf_map_saved_path, exist_ok=True)

        self.test_data_loader = make_data_loader(args)

    def infer(self):
        print('---------Starting Inference-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        torch.cuda.empty_cache()

        with torch.no_grad():
            for data in tqdm(self.test_data_loader, desc="Inference"):
                pre_imgs, labels_loc, labels_clf, texts_data, names = data

                pre_imgs = pre_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()

                output_loc, output_clf = self.deep_model(pre_imgs, texts_data)

                preds_loc = torch.argmax(output_loc, dim=1).cpu().numpy()
                labels_loc_np = labels_loc.cpu().numpy()

                preds_clf = torch.argmax(output_clf, dim=1).cpu().numpy()
                labels_clf_np = labels_clf.cpu().numpy()

                if self.args.buffer is not None and 0.0 < self.args.buffer <= 1.0:
                    H, W = preds_loc.shape[1], preds_loc.shape[2]
                    crop_h, crop_w = int(H * self.args.buffer), int(W * self.args.buffer)
                    start_h, start_w = (H - crop_h) // 2, (W - crop_w) // 2

                    eval_preds_loc = preds_loc[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
                    eval_labels_loc = labels_loc_np[:, start_h:start_h+crop_h, start_w:start_w+crop_w]

                    eval_preds_clf = preds_clf[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
                    eval_labels_clf = labels_clf_np[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
                else:
                    eval_preds_loc = preds_loc
                    eval_labels_loc = labels_loc_np
                    eval_preds_clf = preds_clf
                    eval_labels_clf = labels_clf_np

                self.evaluator_loc.add_batch(eval_labels_loc, eval_preds_loc)

                valid_mask = eval_labels_clf != 255
                self.evaluator_clf.add_batch(eval_labels_clf[valid_mask], eval_preds_clf[valid_mask])

                if self.args.save_images:
                    if self.args.buffer is not None and self.args.save_mode == 'cropped':
                        save_preds_loc = eval_preds_loc
                        save_preds_clf = eval_preds_clf
                    else:
                        save_preds_loc = preds_loc
                        save_preds_clf = preds_clf

                    for i in range(save_preds_loc.shape[0]):
                        base_name = names[i].replace('.json', '').replace('.png', '').replace('.tif', '')
                        png_name = base_name + '.png'
                        tif_name = base_name + '.tif'

                        out_loc_img = save_preds_loc[i].astype(np.uint8)
                        out_clf_img = save_preds_clf[i].astype(np.uint8)

                        # 255 means no building; 0 means no damage.
                        out_clf_img[out_loc_img == 0] = 255

                        imageio.imwrite(os.path.join(self.clf_map_saved_path, png_name), out_clf_img)

                        gt_tif_path = os.path.join(self.args.ref_tif_dir, tif_name)

                        if os.path.exists(gt_tif_path):
                            with rasterio.open(gt_tif_path, 'r') as src:
                                if self.args.buffer is not None and self.args.save_mode == 'cropped':
                                    window = Window(col_off=start_w, row_off=start_h, width=crop_w, height=crop_h)
                                    transform = src.window_transform(window)
                                    height, width = crop_h, crop_w
                                else:
                                    transform = src.transform
                                    height, width = src.height, src.width

                                with rasterio.open(
                                    os.path.join(self.clf_map_saved_path, tif_name),
                                    'w',
                                    driver='GTiff',
                                    height=height,
                                    width=width,
                                    count=1,
                                    dtype=rasterio.uint8,
                                    crs=src.crs,
                                    transform=transform,
                                ) as dst:
                                    dst.write(out_clf_img, 1)
                        else:
                            print(f"\nWarning: Ground truth TIF not found at {gt_tif_path}. Skipping TIF generation for {tif_name}.")

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()

        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / (damage_f1_score + 1e-8))
        oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

        print('\n---------Inference Results-----------')
        if self.args.buffer is not None:
            print(f'Evaluated on center {self.args.buffer * 100}% of images.')
        print(f'Building Localization F1: {loc_f1_score:.4f}')
        print(f'Damage Classification Harmonic Mean F1: {harmonic_mean_f1:.4f}')
        print(f'Overall F1 (oaF1): {oaf1:.4f}')
        print(f'Sub-class F1 Scores: {damage_f1_score}')
        print('-------------------------------------')


def main():
    parser = argparse.ArgumentParser(description="Inference on TextRSDataset with VisionLanguageSSR")

    parser.add_argument('--dataset', type=str, default='TextRSDataset')
    parser.add_argument('--dataset_path', type=str, default='../data')
    parser.add_argument('--data_list_path', type=str, default='../data/splits/Test_dataset.txt')

    parser.add_argument('--ref_tif_dir', type=str, default='../data/Satellite_Imagery',
                        help="Directory containing original .tif files to copy spatial metadata.")

    parser.add_argument('--model_size', type=str, default='Large',
                        choices=['Small', 'Base', 'Large', '7B'])
    parser.add_argument('--backbone_weights', type=str,
                        default='models/dinov3/dinov3/pretrained_weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
                        help='Path to the DINOv3 backbone pretrained weights')
    parser.add_argument('--geo_embed_type', type=str, default='RoPE',
                        choices=['RoPE', 'RFF', 'none'])
    parser.add_argument('--ablated_vision', action='store_true',
                        help="If set, replace DINOv3 features with zero tensors")

    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_loc_classes', type=int, default=2)
    parser.add_argument('--resume', type=str, required=True,
                        help="Path to the trained model checkpoint (.pth)")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--buffer', type=float, default=None,
                        help="Float 0-1: if set, evaluate F1 only on center xx%% of output.")
    parser.add_argument('--crop_size', type=int, default=1024)

    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--save_mode', type=str, default='full', choices=['full', 'cropped'])
    parser.add_argument('--result_saved_path', type=str, default='./results')

    args = parser.parse_args()

    args.model_tag = build_model_tag(args)

    if not os.path.exists(args.data_list_path):
        raise FileNotFoundError(f"Data list not found: {args.data_list_path}")

    with open(args.data_list_path, "r") as f:
        args.data_name_list = [name.strip() for name in f]
        print(f"Number of inference samples found: {len(args.data_name_list)}")
        print(f"Sample data names: {args.data_name_list[:5]} ...")

    inferencer = Inferencer(args)
    inferencer.infer()


if __name__ == "__main__":
    main()
