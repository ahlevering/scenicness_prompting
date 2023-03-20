from PIL import Image
import csv
from sklearn.model_selection import KFold

from pathlib import Path

def load_csv(fpath):
    with open(fpath, 'r', newline='') as csv_file:
        f = csv.reader(csv_file, quoting=csv.QUOTE_ALL)
        rows = [row for row in f]#[0]
        # rows = [int(r) for r in rows] # Fix types
        return rows

def make_crossval_splits(labels, n_crossval):
    kf = KFold(n_splits=n_crossval, shuffle=True, random_state=113)
    all_splits = [k for k in kf.split(labels)]
    train_indices = [k[0] for k in all_splits]
    val_indices = [k[1] for k in all_splits]
    
    # Get original id numbers back from id list
    train_indices = [[labels[k] for k in bin_splits] for bin_splits in train_indices]
    val_indices = [[labels[k] for k in bin_splits] for bin_splits in val_indices]
    return train_indices, val_indices

def recursively_get_files(base_dir, extensions=['.tif', '.jpg', '.png', '.tiff']):
    file_list = []
    for filetype in extensions:
        file_list.extend([str(img_path) for img_path in Path(base_dir).glob('**/*{}'.format(filetype))])
    return file_list

def get_image_label_pairs(images_root, labels_root):
    images = recursively_get_files(images_root)
    labels = recursively_get_files(labels_root, ['.geojson'])

    img_label_pairs = []
    for img in images:
        # Remove extension
        img_city = Path(img).stem.split('_')[0]
        label_file = [l for l in labels if img_city in Path(l).stem.split('_')[0]][0]
        img_label_pairs.append([img, label_file])
    return img_label_pairs

def overlay_images(orig, overlay, opacity=0.5):
    # TODO: NORMALIZE, COLOUR TRANSFORM
    orig = orig.convert("RGBA")
    overlay_resampled = overlay.resize(orig.size, resample=Image.BICUBIC)
    overlay_resampled = overlay_resampled.convert("RGBA")
    overlaid = Image.blend(orig, overlay_resampled, opacity)
    return overlaid