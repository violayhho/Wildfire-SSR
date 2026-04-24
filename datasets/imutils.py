import random
import numpy as np


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    img_array = np.asarray(img, dtype=np.float32)
    return (img_array - mean) / std


def random_fliplr(*arrays):
    if random.random() > 0.5:
        arrays = tuple(np.fliplr(array) for array in arrays)
    return arrays


def random_flipud(*arrays):
    if random.random() > 0.5:
        arrays = tuple(np.flipud(array) for array in arrays)
    return arrays


def random_rot(*arrays):
    k = random.randrange(4)
    if k != 0:
        arrays = tuple(np.rot90(array, k).copy() for array in arrays)
    return arrays


def random_crop(*images, crop_size, mean=None, ignore_index=255, **labels):
    if mean is None:
        mean = [0] * sum([img.shape[2] for img in images])

    h, w = list(labels.values())[0].shape
    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_images = []
    for img in images:
        pad_image = np.zeros((H, W, img.shape[2]), dtype=np.float32)
        for i in range(img.shape[2]):
            pad_image[:, :, i] = mean[i]
        pad_images.append(pad_image)

    pad_labels = {}
    for key in labels:
        pad_labels[key] = np.ones((H, W), dtype=np.float32) * ignore_index

    H_pad = np.random.randint(H - h + 1)
    W_pad = np.random.randint(W - w + 1)

    for idx, img in enumerate(images):
        pad_images[idx][H_pad:H_pad + h, W_pad:W_pad + w, :] = img

    for key, label in labels.items():
        pad_labels[key][H_pad:H_pad + h, W_pad:W_pad + w] = label

    def get_random_cropbox(cat_max_ratio=0.75):
        for _ in range(10):
            H_start = random.randrange(0, H - crop_size + 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1)
            W_end = W_start + crop_size

            temp_label = pad_labels[list(labels.keys())[0]][H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()

    cropped_images = [img[H_start:H_end, W_start:W_end, :] for img in pad_images]
    cropped_labels = {key: label[H_start:H_end, W_start:W_end] for key, label in pad_labels.items()}

    return (*cropped_images, *cropped_labels.values())


def random_fliplr_with_points(*arrays, texts_with_locations):
    if random.random() > 0.5:
        w = arrays[0].shape[1]
        arrays = tuple(np.fliplr(arr).copy() for arr in arrays)
        for point in texts_with_locations:
            py, px = point['pixel_coord']
            point['pixel_coord'] = (py, w - 1 - px)
    return (*arrays, texts_with_locations)


def random_flipud_with_points(*arrays, texts_with_locations):
    if random.random() > 0.5:
        h = arrays[0].shape[0]
        arrays = tuple(np.flipud(arr).copy() for arr in arrays)
        for point in texts_with_locations:
            py, px = point['pixel_coord']
            point['pixel_coord'] = (h - 1 - py, px)
    return (*arrays, texts_with_locations)


def random_rot_with_points(*arrays, texts_with_locations):
    k = random.randrange(4)
    if k != 0:
        h, w = arrays[0].shape[:2]
        arrays = tuple(np.rot90(arr, k).copy() for arr in arrays)
        for point in texts_with_locations:
            py, px = point['pixel_coord']
            if k == 1:    # 90 degrees CCW
                point['pixel_coord'] = (w - 1 - px, py)
            elif k == 2:  # 180 degrees CCW
                point['pixel_coord'] = (h - 1 - py, w - 1 - px)
            elif k == 3:  # 270 degrees CCW
                point['pixel_coord'] = (px, h - 1 - py)
    return (*arrays, texts_with_locations)


def random_crop_with_points(*images, texts_with_locations, crop_size, mean=None, ignore_index=255, **labels):
    if mean is None:
        mean = [0] * sum([img.shape[2] for img in images])

    h, w = list(labels.values())[0].shape
    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_images = []
    for img in images:
        pad_image = np.zeros((H, W, img.shape[2]), dtype=np.float32)
        for i in range(img.shape[2]):
            pad_image[:, :, i] = mean[i]
        pad_images.append(pad_image)

    pad_labels = {key: np.full((H, W), ignore_index, dtype=np.float32) for key in labels}

    H_pad = np.random.randint(0, H - h + 1) if H > h else 0
    W_pad = np.random.randint(0, W - w + 1) if W > w else 0

    for i, img in enumerate(images):
        pad_images[i][H_pad:H_pad + h, W_pad:W_pad + w] = img

    for key, label in labels.items():
        pad_labels[key][H_pad:H_pad + h, W_pad:W_pad + w] = label

    padded_points = [
        {'text': p['text'], 'pixel_coord': (p['pixel_coord'][0] + H_pad, p['pixel_coord'][1] + W_pad)}
        for p in texts_with_locations
    ]

    def get_random_cropbox(cat_max_ratio=0.75):
        for _ in range(10):
            H_start = random.randrange(0, H - crop_size + 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1)
            W_end = W_start + crop_size

            temp_label = pad_labels[list(labels.keys())[0]][H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()

    cropped_images = [img[H_start:H_end, W_start:W_end, :] for img in pad_images]
    cropped_labels = {key: label[H_start:H_end, W_start:W_end] for key, label in pad_labels.items()}

    points_cropped = []
    for point in padded_points:
        py, px = point['pixel_coord']
        if H_start <= py < H_start + crop_size and W_start <= px < W_start + crop_size:
            points_cropped.append({'text': point['text'], 'pixel_coord': (py - H_start, px - W_start)})

    return (*cropped_images, *cropped_labels.values(), points_cropped)
