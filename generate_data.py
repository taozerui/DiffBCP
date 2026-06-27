import math
import cv2
import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.io import savemat
from tqdm import trange

_FFHQ_ID = [0, 1, 3, 6, 14, 15, 38]  # some popular IDs in previous papers
_FFHQ_PATH = '/work1/data/ffhq-dataset/images1024x1024/00000'
_IMAGENET_PATH = '/work1/data/imagenet-1k/val'
N = 128
SEED = 42


def sample_n_images_to_numpy_ffhq():
    rng = np.random.Generator(np.random.PCG64(SEED))
    id = rng.choice(1000, size=N, replace=False).tolist()
    id = _FFHQ_ID + id
    id = list(set(id))
    id = sorted(id[:N])

    arrs = []
    for i in id:
        img_path = os.path.join(_FFHQ_PATH, f'{i:05d}.png')
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 256), Image.BICUBIC)
        arrs.append(np.array(img))
    return np.stack(arrs)


def sample_n_images_to_numpy_imagenet():
    rng = np.random.Generator(np.random.PCG64(SEED + 10))

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    images = []
    for root, _, files in os.walk(_IMAGENET_PATH):
        for f in files:
            if os.path.splitext(f)[1].lower() in image_extensions:
                images.append(os.path.join(root, f))
    
    n_images = len(images)
    id = rng.choice(n_images, size=N, replace=False).tolist()
    selected_images = [images[i] for i in id]

    arrs = []
    for img_path in selected_images:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 256), Image.BICUBIC)
        arrs.append(np.array(img))
    return np.stack(arrs)


def random_mask(x, obs_ratio: float):
    rng = np.random.Generator(np.random.PCG64(SEED + 20))
    mask = np.zeros(x.shape).reshape(-1)
    n = int(obs_ratio * x.size)
    idx = rng.permutation(x.size)[:n]
    mask[idx] = 1
    return mask.reshape(x.shape)


def random_stripe_mask(x, row_mask_ratio=0.3, col_mask_ratio=0.3, max_missing_pixel=None):
    rng = np.random.Generator(np.random.PCG64(SEED + 22))
    
    bs, h, w, c = x.shape
    
    mask = np.ones_like(x)
    
    for i in range(bs):
        # Generate row stripes
        if row_mask_ratio > 0:
            total_row_pixels = int(h * row_mask_ratio)
            
            max_stripe_width = max(1, h // 4)
            if max_missing_pixel is not None:
                max_stripe_width = min(max_stripe_width, max_missing_pixel)
            row_stripes = []
            remaining_pixels = total_row_pixels
            
            while remaining_pixels > 0:
                stripe_width = min(remaining_pixels, int(rng.integers(1, max_stripe_width + 1)))
                stripe_start = int(rng.integers(0, h - stripe_width + 1))
                row_stripes.append((stripe_start, stripe_start + stripe_width))
                remaining_pixels -= stripe_width
                
                if len(row_stripes) > h:
                    break
            
            for start_row, end_row in row_stripes:
                mask[i, start_row:end_row, :, :] = 0
        
        # Generate column stripes
        if col_mask_ratio > 0:
            total_col_pixels = int(w * col_mask_ratio)
            
            max_stripe_width = max(1, w // 4)
            if max_missing_pixel is not None:
                max_stripe_width = min(max_stripe_width, max_missing_pixel)
            col_stripes = []
            remaining_pixels = total_col_pixels
            
            while remaining_pixels > 0:
                stripe_width = min(remaining_pixels, int(rng.integers(1, max_stripe_width + 1)))
                stripe_start = int(rng.integers(0, w - stripe_width + 1))
                col_stripes.append((stripe_start, stripe_start + stripe_width))
                remaining_pixels -= stripe_width
                
                if len(col_stripes) > w:
                    break
            
            for start_col, end_col in col_stripes:
                mask[i, :, start_col:end_col, :] = 0
    
    return mask


def random_irregular_mask(
    img_shape,
    num_vertices=(4, 8),
    max_angle=4,
    length_range=(10, 100),
    brush_width=(10, 40),
    dtype='uint8',
    seed=0,
):
    """Generate random irregular masks.
    Source: https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/data/util/mask.py

    This is a modified version of free-form mask implemented in
    'brush_stroke_mask'.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 8).
        max_angle (float): Max value of angle at each vertex. Default 4.0.
        length_range (int | tuple[int]): (min_length, max_length). If only give
            an integer, we will fix the length of brush. Default: (10, 100).
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (10, 40).
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'
        seed (int, optional): Random seed for reproducibility. Default: None.

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """
    np.random.seed(seed)

    h, w = img_shape[:2]

    mask = np.zeros((h, w), dtype=dtype)
    if isinstance(length_range, int):
        min_length, max_length = length_range, length_range + 1
    elif isinstance(length_range, tuple):
        min_length, max_length = length_range
    else:
        raise TypeError('The type of length_range should be int'
                        f'or tuple[int], but got type: {length_range}')
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, int):
        min_brush_width, max_brush_width = brush_width, brush_width + 1
    elif isinstance(brush_width, tuple):
        min_brush_width, max_brush_width = brush_width
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    num_v = np.random.randint(min_num_vertices, max_num_vertices)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        # from the start point, randomly setlect n \in [1, 6] directions.
        direction_num = np.random.randint(1, 6)
        angle_list = np.random.randint(0, max_angle, size=direction_num)
        length_list = np.random.randint(
            min_length, max_length, size=direction_num)
        brush_width_list = np.random.randint(
            min_brush_width, max_brush_width, size=direction_num)
        for direct_n in range(direction_num):
            angle = 0.01 + angle_list[direct_n]
            if i % 2 == 0:
                angle = 2 * math.pi - angle
            length = length_list[direct_n]
            brush_w = brush_width_list[direct_n]
            # compute end point according to the random angle
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_w)
            start_x, start_y = end_x, end_y
    mask = np.expand_dims(mask, axis=2)

    return mask


def get_irregular_mask(x, area_ratio_range=(0.15, 0.5), **kwargs):
    """Get irregular mask with the constraints in mask ratio
    Source: https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/data/util/mask.py

    Args:
        img_shape (tuple[int]): Size of the image.
        area_ratio_range (tuple(float)): Contain the minimum and maximum area
        ratio. Default: (0.15, 0.5).
        seed (int, optional): Random seed for reproducibility. Default: None.

    Returns:
        numpy.ndarray: Mask in the shape of (N, h, w, 3).
    """
    np.random.seed(SEED + 24)
    
    n, h, w, c = x.shape
    dtype = x.dtype
    mask = np.zeros(x.shape)
    img_shape = (h, w)

    _seed = SEED + 1000

    for i in trange(n):
        mask_i = random_irregular_mask(img_shape, seed=_seed, **kwargs)
        mask_i = 1 - mask_i
        _seed += 1
        min_ratio, max_ratio = area_ratio_range

        mask_ratio = np.sum(mask_i) / (img_shape[0] * img_shape[1])

        while mask_ratio < min_ratio:
            mask_i = random_irregular_mask(img_shape, seed=_seed, **kwargs)
            mask_i = 1 - mask_i
            mask_ratio = np.sum(mask_i) / (img_shape[0] * img_shape[1])
            _seed += 1
        while mask_ratio > max_ratio:
            mask_i_second = random_irregular_mask(img_shape, seed=_seed, **kwargs)
            mask_i_second = 1 - mask_i_second
            mask_i = (mask_i == 1) & (mask_i_second == 1)
            mask_i = mask_i.astype(dtype)
            mask_ratio = np.sum(mask_i) / (img_shape[0] * img_shape[1])
            _seed += 1

        # while not min_ratio < (np.sum(mask_i) /
        #                     (img_shape[0] * img_shape[1])) < max_ratio:
        #     mask_i = random_irregular_mask(img_shape, seed=_seed, **kwargs)
        #     _seed += 1
        mask_i = np.repeat(mask_i, c, axis=2).astype(dtype)
        mask[i] = mask_i

    return mask


arr = np.ones([N, 256, 256, 3])
generating_mask = {'random', 'stripe', 'irregular'}

Path('./data').mkdir(parents=True, exist_ok=True)

if 'random' in generating_mask:
    mask = random_mask(arr, obs_ratio=0.3)
    np.save(f'./data/random_mask_obs03.npy', mask)
    savemat(f'./data/random_mask_obs03.mat', {'mask': mask})
    mask = random_mask(arr, obs_ratio=0.1)
    np.save(f'./data/random_mask_obs01.npy', mask)
    savemat(f'./data/random_mask_obs01.mat', {'mask': mask})

# generate stripe masks
if 'stripe' in generating_mask:
    stripe_mask = random_stripe_mask(arr, row_mask_ratio=0.3, col_mask_ratio=0.3, max_missing_pixel=8)
    np.save(f'./data/stripe_mask_row03_col03.npy', stripe_mask)
    savemat(f'./data/stripe_mask_row03_col03.mat', {'mask': stripe_mask})

# generate irregular masks
if 'irregular' in generating_mask:
    irregular_mask = get_irregular_mask(arr, area_ratio_range=(0.50, 0.70), brush_width=(2, 6), length_range=(10, 200))
    np.save(f'./data/irregular_mask_area50-70_brush2-6.npy', irregular_mask)
    savemat(f'./data/irregular_mask_area50-70_brush2-6.mat', {'mask': irregular_mask})

    arr = np.ones([1, 2048, 2048, 3])
    irregular_mask = get_irregular_mask(arr, area_ratio_range=(0.50, 0.70), brush_width=(2, 6), length_range=(10, 200))
    np.save(f'./data/putt_irregular_mask_area50-70_brush2-6.npy', irregular_mask)
    savemat(f'./data/putt_irregular_mask_area50-70_brush2-6.mat', {'mask': irregular_mask})
