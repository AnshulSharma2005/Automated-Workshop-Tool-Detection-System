import os
import shutil

DATASET_DIR = r"data/ToolsFixed"  # Change if needed

def fix_yolo_label_file(label_path, img_w, img_h):
    fixed_lines = []
    changed = False

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()

            # Must be at least 5 elements: cls cx cy w h
            if len(parts) < 5:
                print(f"❌ BAD LABEL (too few values) → {label_path}")
                changed = True
                continue

            cls, cx, cy, w, h = parts[:5]

            try:
                cx, cy, w, h = map(float, [cx, cy, w, h])
            except:
                print(f"❌ NON-NUMERIC LABEL → {label_path}")
                changed = True
                continue

            # Remove boxes outside range
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                print(f"❌ INVALID VALUES → {label_path} : {line.strip()}")
                changed = True
                continue

            fixed_lines.append(f"{cls} {cx} {cy} {w} {h}\n")

    # If empty, delete file
    if len(fixed_lines) == 0:
        print(f"⚠️ EMPTY LABEL REMOVED → {label_path}")
        os.remove(label_path)
        return

    # Save fixed file
    if changed:
        print(f"✔ FIXED → {label_path}")
        with open(label_path, "w") as f:
            f.writelines(fixed_lines)


def process_split(split):
    print(f"\n===== FIXING {split.upper()} =====")

    images_dir = os.path.join(DATASET_DIR, split, "images")
    labels_dir = os.path.join(DATASET_DIR, split, "labels")

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, img_name.rsplit('.', 1)[0] + ".txt")

        # Remove images with no labels
        if not os.path.exists(label_path):
            print(f"❌ NO LABEL — REMOVED IMAGE → {img_path}")
            os.remove(img_path)
            continue

        # Read image size
        try:
            from PIL import Image
            img = Image.open(img_path)
            w, h = img.size
        except:
            print(f"❌ BAD IMAGE REMOVED → {img_path}")
            os.remove(img_path)
            continue

        # Fix its label file
        fix_yolo_label_file(label_path, w, h)


def create_yaml():
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    content = """train: train/images
val: valid/images
test: test/images

nc: 7
names: ['Files', 'Hammers', 'Pincers', 'Pliers', 'Scissors', 'Screwdriver', 'Spanners']
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"\n✔ data.yaml rebuilt → {yaml_path}")


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        process_split(split)

    create_yaml()
    print("\n🎉 Dataset cleaning complete!")
