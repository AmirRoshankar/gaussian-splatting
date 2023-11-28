import cv2
import numpy as np
import os
import logging
from argparse import ArgumentParser
import shutil
import json
import random
import pathlib

from tqdm import tqdm

def refresh_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def convert_backgrounds(img_dir, mask_dir, dst_dir):
    for f in os.listdir(img_dir):
        if not f.endswith(".png"):
            continue
        
        # Construct the corresponding file paths for image and mask
        image_path = os.path.join(img_dir, f)
        mask_file = f.replace("frame", "mask")
        mask_path = os.path.join(mask_dir, mask_file)
        dst_path = os.path.join(dst_dir, f)

        # Load your image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Create a 4-channel blank image with transparency (alpha channel)
        height, width = mask.shape
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

        # Set the RGB channels of the RGBA image to the original image
        rgba_image[:, :, :3] = image

        # Set the alpha channel of the RGBA image to the binary mask
        _, bin_mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
        rgba_image[:, :, 3] = bin_mask

        # Save the resulting masked image as a PNG with a transparent background
        cv2.imwrite(dst_path, rgba_image)

def create_splits(input_dir, train_dir, test_dir, test_split=0.15):
    refresh_dir(train_dir)
    refresh_dir(test_dir)

    file_paths = [f for f in os.listdir(input_dir)]
    random.shuffle(file_paths)
    num_test = int(len(file_paths) * test_split)
    train = file_paths[num_test:]
    test = file_paths[:num_test]

    for f in test:
        shutil.move(os.path.join(input_dir, f), os.path.join(test_dir, f))
    
    for f in train:
        shutil.move(os.path.join(input_dir, f), os.path.join(train_dir, f))
    
    return train, test

def convert_transforms(all_tforms_path, test, dst_path):
    all_tforms = open(all_tforms_path)
    all_tforms = all_tforms.read()
    all_tforms = json.loads(all_tforms)

    test = set(test)

    camera_angle_x = 0.6911112070083618
    # rotation = 0.012566370614359171
    volume_shape = all_tforms['volume_shape']
    volume_spacing = all_tforms['volume_spacing']

    train_frames = []
    test_frames = []
    for f in all_tforms['frames']:
        transform_matrix = f['transform_matrix']
        slice_range = [0,-1]
        fpath = f['file_path']
        fname = fpath[fpath.index('/') + 1:]

        if fname in test:
            # test sample
            fpath = f"./test/{fname[:-4]}"
            mpath = f"./masks/{fname[:-4].replace('frame', 'mask')}"
            test_frames.append({
                "file_path": fpath,
                "mask_path": mpath,
                "slice_range": slice_range,
                "transform_matrix": transform_matrix,
            })
        else:
            # train sample
            fpath = f"./train/{fname[:-4]}"
            # mpath = f"./train/masks/mask_{fname[:-4].split('_')[1]}"
            mpath = f"./masks/{fname[:-4].replace('frame', 'mask')}"
            train_frames.append({
                "file_path": fpath,
                "mask_path": mpath,
                "slice_range": slice_range,
                "transform_matrix": transform_matrix,
            })

    transforms_train = {"camera_angle_x": camera_angle_x,
                        "volume_shape": volume_shape,
                        "volume_spacing": volume_spacing,
                        "frames": train_frames}

    transforms_test = {"camera_angle_x": camera_angle_x,
                        "volume_shape": volume_shape,
                        "volume_spacing": volume_spacing,
                        "frames": test_frames}

    transforms_train_path = os.path.join(dst_path, "transforms_train.json")
    transforms_test_path = os.path.join(dst_path, "transforms_test.json")

    with open(transforms_test_path, "w") as outfile: 
        json.dump(transforms_test, outfile, indent=4)
    
    with open(transforms_train_path, "w") as outfile: 
        json.dump(transforms_train, outfile, indent=4)

def convert_transforms_as(src_path, dst_path):
    raw_tforms = open(src_path)
    raw_tforms = raw_tforms.read()
    raw_tforms = json.loads(raw_tforms)

    camera_angle_x = 0.6911112070083618
    volume_shape = raw_tforms['volume_shape']
    volume_spacing = raw_tforms['volume_spacing']

    subdir = pathlib.PurePath(src_path).parent.name

    frames = []
    for f in raw_tforms['frames']:
        transform_matrix = f['transform_matrix']
        slice_range = [0,-1]
        fpath = f['file_path']
        fname = fpath[fpath.index('/') + 1:]

        # test sample
        fpath = f"./{subdir}/{fname[:-4]}"
        # mpath = f"./{subdir}/masks/mask_{fname[:-4].split('_')[1]}"
        mpath = f"./masks/{fname[:-4].replace('frame', 'mask')}"
        frames.append({
            "file_path": fpath,
            "mask_path": mpath,
            "slice_range": slice_range,
            "transform_matrix": transform_matrix,
        })

    new_tforms = {"camera_angle_x": camera_angle_x,
                        "volume_shape": volume_shape,
                        "volume_spacing": volume_spacing,
                        "frames": frames}


    with open(dst_path, "w") as outfile: 
        json.dump(new_tforms, outfile, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser("Preprocessing")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    args = parser.parse_args()

    # iterate over each layer and the composite
    for dir in tqdm(os.listdir(args.source_path)):
        layer_src_path = os.path.join(args.source_path, dir)
        if not os.path.isdir(layer_src_path):
            continue
        
        img_dir = os.path.join(layer_src_path, "images") # copy over from mednerf
        mask_dir = os.path.join(layer_src_path, "masks")  # copy over from mednerf
        transparent_dir = os.path.join(layer_src_path, "input")
        train_dir = os.path.join(layer_src_path, "train")
        test_dir = os.path.join(layer_src_path, "test")
        all_tform_path = os.path.join(layer_src_path, "transforms.json")

        if os.path.exists(transparent_dir):
            shutil.rmtree(transparent_dir)
        os.makedirs(transparent_dir)

        # Convert images to 4 channel with transparent backgrounds
        convert_backgrounds(img_dir, mask_dir, transparent_dir)

        # Create splits
        train, test = create_splits(transparent_dir, train_dir, test_dir, test_split=0.15)
        shutil.rmtree(transparent_dir)

        # Create transform jsons
        convert_transforms(all_tform_path, test, layer_src_path)