import os
import nibabel as nib
import numpy as np

def check_dataset(images_dir, labels_dir, min_shape=(8, 8, 8)):
    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    masks = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])

    assert len(images) == len(masks), "Nombre d'images et de masques différent !"

    for img_path, mask_path in zip(images, masks):
        print(f"Vérification : {os.path.basename(img_path)} / {os.path.basename(mask_path)}")
        try:
            img = nib.load(img_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()
        except Exception as e:
            print(f"❌ Erreur de lecture : {e}")
            continue

        if img.shape != mask.shape:
            print(f"❌ Dimensions différentes : image {img.shape}, masque {mask.shape}")
            continue

        if any(s < ms for s, ms in zip(img.shape, min_shape)):
            print(f"❌ Volume trop petit : {img.shape}, requis min {min_shape}")
            continue

        unique_mask = np.unique(mask)
        if not np.all(np.isin(unique_mask, [0, 1])):
            print(f"⚠️ Masque non binaire : valeurs trouvées {unique_mask}")

        print("✅ OK")

if __name__ == "__main__":
    images_dir = "./dataset/images"
    labels_dir = "./dataset/labels"
    check_dataset(images_dir, labels_dir, min_shape=(8, 64, 64))  # adapte min_shape à ton modèle
