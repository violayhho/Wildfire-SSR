import argparse
import os
import json
import imageio
import rasterio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import datasets.imutils as imutils
from datasets.json_to_nl import build_clip_caption

DINOV3_SAT_493M_MEAN = [0.430, 0.411, 0.296]
DINOV3_SAT_493M_STD = [0.213, 0.156, 0.143]

def img_loader(path):
    if '.tif' in path:
        with rasterio.open(path, 'r') as src:
            img = src.read().transpose([1, 2, 0])
    else:
        img = imageio.imread(path)
    return img


class TextRSDataset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_epochs=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.crop_size = crop_size

        if max_epochs is not None:
            self.data_list = self.data_list * max_epochs

        self.image_dir = os.path.join(self.dataset_path, 'Satellite_Imagery')
        self.loc_label_dir = os.path.join(self.dataset_path, 'Building_Footprints')
        self.clf_label_dir = os.path.join(self.dataset_path, 'Wildfire_Risk_Labels')
        self.text_dir = os.path.join(self.dataset_path, 'Street_View_Text')
    
    def _convert_pixel_to_normalized_coord(self, texts_with_locations, img_h, img_w):
        if not texts_with_locations:
            return []
            
        for item in texts_with_locations:
            pixel_y, pixel_x = item['pixel_coord']
            
            normalized_x = pixel_x / img_w
            normalized_y = pixel_y / img_h
            
            # Clamp to [0, 1] to handle any slight float errors or edge-cropping
            normalized_x = min(max(normalized_x, 0.0), 1.0)
            normalized_y = min(max(normalized_y, 0.0), 1.0)
            
            item['normalized_coord'] = (normalized_x, normalized_y)
            
            del item['pixel_coord']

        return texts_with_locations

    def __transforms(self, aug, pre_img, loc_label, clf_label, texts_with_locations):
        if aug:
            pre_img, loc_label, clf_label, texts_with_locations = imutils.random_crop_with_points(
                pre_img, texts_with_locations = texts_with_locations, crop_size = self.crop_size, \
                mean=np.array(DINOV3_SAT_493M_MEAN) * 255, \
                loc_label=loc_label, clf_label=clf_label)

            pre_img[np.isnan(pre_img)] = 0
            loc_label[np.isnan(loc_label)] = 255
            loc_label[np.all(pre_img == [255, 255, 255], axis=-1)] = 255
            clf_label[np.isnan(clf_label)] = 255
            clf_label[np.all(pre_img == [0, 0, 0], axis=-1)] = 255
            
            pre_img, loc_label, clf_label, texts_with_locations = imutils.random_fliplr_with_points(
                pre_img, loc_label, clf_label, texts_with_locations= texts_with_locations
            )
            pre_img, loc_label, clf_label, texts_with_locations = imutils.random_flipud_with_points(
                pre_img, loc_label, clf_label, texts_with_locations = texts_with_locations
            )

            pre_img, loc_label, clf_label, texts_with_locations = imutils.random_rot_with_points(
                pre_img, loc_label, clf_label, texts_with_locations = texts_with_locations
            )

        pre_img = pre_img.astype(np.float32) / 255.0
        pre_img = imutils.normalize_img(pre_img, mean=DINOV3_SAT_493M_MEAN, std=DINOV3_SAT_493M_STD)
        pre_img = pre_img.astype(np.float32)
        pre_img = np.transpose(pre_img, (2, 0, 1))

        return pre_img, loc_label, clf_label, texts_with_locations

    def __getitem__(self, index):
        data_name = self.data_list[index]
        img_path = os.path.join(self.image_dir, data_name)
        loc_label_path = os.path.join(self.loc_label_dir, data_name)
        clf_label_path = os.path.join(self.clf_label_dir, data_name)

        pre_img = self.loader(img_path)
        
        if os.path.exists(loc_label_path):
            loc_label = self.loader(loc_label_path)
            if loc_label.ndim == 3:
                loc_label = loc_label[:, :, 0]
        else:
            loc_label = np.ones_like(pre_img[:, :, 0]) * 255

        if os.path.exists(clf_label_path):
            clf_label = self.loader(clf_label_path)
            if clf_label.ndim == 3:
                clf_label = clf_label[:, :, 0]
        else:
            clf_label = np.ones_like(pre_img[:, :, 0]) * 255

        pre_img[np.isnan(pre_img)] = 255
        pre_img[np.all(pre_img == [0, 0, 0], axis=-1)] = [255, 255, 255]

        loc_label[np.isnan(loc_label)] = 255
        loc_label[np.all(pre_img == [255, 255, 255], axis=-1)] = 255
        clf_label[np.isnan(clf_label)] = 255
        clf_label[np.all(pre_img == [255, 255, 255], axis=-1)] = 255


        json_name = os.path.splitext(data_name)[0] + '.json'
        text_path = os.path.join(self.text_dir, json_name)

        texts_with_locations = []
        
        if os.path.exists(text_path):
            with open(text_path, 'r') as f:
                text_data = json.load(f)
                for item in text_data:
                    text_dict = item['text'][0] if isinstance(item['text'], list) else item['text']
                    texts_with_locations.append({
                        'text': build_clip_caption(text_dict),
                        'pixel_coord': tuple(item['pixel_coord']),
                    })

        aug = 'train' in self.type
        pre_img, loc_label, clf_label, texts_with_locations = self.__transforms(
            aug, pre_img, loc_label, clf_label, texts_with_locations
        )
        if not aug:
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)
        
        current_h = pre_img.shape[1]
        current_w = pre_img.shape[2]
        
        texts_with_locations = self._convert_pixel_to_normalized_coord(
            texts_with_locations, 
            img_h=current_h, 
            img_w=current_w
        )
        
        return pre_img, loc_label, clf_label, texts_with_locations, data_name

    def __len__(self):
        return len(self.data_list)

def text_rs_collate_fn(batch):
    """
    Custom collate function to handle batches of images, labels, and
    variable-length lists of text data.
    """
    # Separate the components of the batch
    images = [torch.from_numpy(item[0]) for item in batch]
    loc_labels = [torch.from_numpy(item[1]) for item in batch]
    clf_labels = [torch.from_numpy(item[2]) for item in batch]
    texts_data = [item[3] for item in batch] # This is a list of lists of dicts
    data_names = [item[4] for item in batch]

    # Stack images and labels as usual
    images = torch.stack(images, 0)
    loc_labels = torch.stack(loc_labels, 0)
    clf_labels = torch.stack(clf_labels, 0)

    # The text data is returned as a list of lists, to be handled by the text encoder
    return images, loc_labels, clf_labels, texts_data, data_names


def make_data_loader(args):
    if args.dataset == 'TextRSDataset':
        # make sure to use the self-defined collate_fn to handle the text data
        dataset = TextRSDataset(args.dataset_path, args.data_name_list, args.crop_size, args.max_epochs, args.type)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                          collate_fn=text_rs_collate_fn, num_workers=16, drop_last=False)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TextRSDataset DataLoader Test")
    parser.add_argument('--dataset', type=str, default='TextRSDataset')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='../data')
    parser.add_argument('--data_list_path', type=str, default='../data/splits/Train_dataset.txt')
    parser.add_argument('--shuffle', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_name_list', type=list)

    args = parser.parse_args()

    with open(args.data_list_path, "r") as f:
        args.data_name_list = [data_name.strip() for data_name in f]
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        pre_img, loc_label, clf_label, texts_data, _ = data
        texts_per_sample = [len(t) for t in texts_data]
        print(i, "pre_img: ", pre_img.data.size(),
              "loc_label: ", loc_label.data.size(),
              "clf_label: ", clf_label.data.size(),
              "texts_data: ", f"batch={len(texts_data)}, texts_per_sample={texts_per_sample}")
