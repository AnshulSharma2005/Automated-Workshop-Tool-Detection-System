# src/setup_data.py
"""
Prepare dataset folder structure expected by training scripts.
Usage:
    python src/setup_data.py --src data/Tools.v1i.yolov8 --out data
"""

import argparse
import os
import shutil

def copy_tree(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        print(f"{dst_dir} already exists — skipping copy.")
        return
    shutil.copytree(src_dir, dst_dir)
    print(f"Copied {src_dir} -> {dst_dir}")

def make_structure(base_out):
    os.makedirs(os.path.join(base_out, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_out, 'images', 'valid'), exist_ok=True)
    os.makedirs(os.path.join(base_out, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(base_out, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_out, 'labels', 'valid'), exist_ok=True)
    os.makedirs(os.path.join(base_out, 'labels', 'test'), exist_ok=True)

def create_data_yaml(out_path):
    content = """train: data/images/train
val: data/images/valid
test: data/images/test

nc: 7
names: ['Files','Hammers','Pincers','Pliers','Scissors','Screwdriver','Spanners']
"""
    with open(os.path.join(out_path, 'data.yaml'), 'w') as f:
        f.write(content)
    print("Wrote data/data.yaml")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source exported folder (Tools.v1i.yolov8)')
    parser.add_argument('--out', default='data', help='Destination data folder (created/updated)')
    args = parser.parse_args()

    src = args.src.rstrip('/\\')
    out = args.out.rstrip('/\\')

    # create canonical structure
    make_structure(out)

    # if Roboflow exported structure exists at src, copy images & labels into data/
    for split in ('train','valid','test'):
        src_images = os.path.join(src, split, 'images')
        src_labels = os.path.join(src, split, 'labels')
        dst_images = os.path.join(out, 'images', split if split!='valid' else 'valid')
        dst_labels = os.path.join(out, 'labels', split if split!='valid' else 'valid')

        if os.path.exists(src_images):
            for f in os.listdir(src_images):
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    shutil.copy2(os.path.join(src_images, f), os.path.join(out, 'images', split if split!='valid' else 'valid', f))
        if os.path.exists(src_labels):
            for f in os.listdir(src_labels):
                if f.lower().endswith('.txt'):
                    shutil.copy2(os.path.join(src_labels, f), os.path.join(out, 'labels', split if split!='valid' else 'valid', f))

    # write data.yaml
    create_data_yaml(out)
    print("Dataset setup complete. Please verify data/ folder.")

if __name__ == "__main__":
    main()
