import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pathlib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------- Dataset 3D -------------
class Medical3DDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Normalisation image
        img = (img - img.min()) / (img.max() - img.min())

        # Binarisation du masque
        mask = (mask > 0).astype(np.float32)  # ou adapte le seuil selon tes masques

        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)  # (D,H,W)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2,0,1)

        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

# --------- 3D U-Net --------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
    # def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64]):  # U-Net simplifié
        super(UNet3D, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder
        for feature in features:
            self.downs.append(self.conv_block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.conv_block(features[-1], features[-1]*2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self.conv_block(feature*2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.functional.max_pool3d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)

        return torch.sigmoid(self.final_conv(x))

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

# --------- Dice score -------------
def dice_score(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

# --------- Prepare data lists ----------
def get_file_lists(data_dir):
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    masks = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
    assert len(images) == len(masks), "Mismatch images and masks count"
    return images, masks

# --------- Training function ----------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc='Training', leave=False)
    for imgs, masks in loop:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return running_loss / len(loader)

# --------- Validation function ----------
def val_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    dice_total = 0.0
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation', leave=False)
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            dice = dice_score(preds, masks)
            running_loss += loss.item()
            dice_total += dice.item()
            loop.set_postfix(loss=loss.item(), dice=dice.item())
    return running_loss / len(loader), dice_total / len(loader)


# --------- Save challenge prediction ----------
def save_challenge_predictions(model, val_dataset, val_imgs, device, save_dir="submission_labels", threshold=0.5):
    """
    Sauvegarde les prédictions du modèle au format attendu pour une soumission challenge :
    - Un fichier .nii.gz par volume, nommé comme l'input d'origine.
    - Les volumes sont binarisés (0/1).
    """
    model.eval()
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, (img, mask) in enumerate(val_dataset):
            img = img.unsqueeze(0).to(device)
            pred = model(img).squeeze().cpu()
            bin_pred = (pred > threshold).float()

            # On suppose que val_imgs[i] contient le chemin du volume d'origine
            orig_name = pathlib.Path(val_imgs[i]).stem  # ex: "patient01"
            affine = nib.load(val_imgs[i]).affine

            # Si la prédiction est (D,H,W), on remet en (H,W,D) pour compatibilité challenge
            if bin_pred.ndim == 3:
                pred_np = bin_pred.permute(1,2,0).numpy()  # D,H,W -> H,W,D
            else:
                pred_np = bin_pred.numpy()

            # Sauvegarde avec le nom d'origine
            nib.save(nib.Nifti1Image(pred_np.astype('uint8'), affine), save_path / f"{orig_name}.nii.gz")

    print(f"✅ Prédictions sauvegardées dans {save_dir} pour soumission challenge.")

def filter_small_volumes(images, masks, min_shape=(8, 64, 64)):
    filtered_images = []
    filtered_masks = []
    for img_path, mask_path in zip(images, masks):
        img = nib.load(img_path).get_fdata()
        if all(s >= ms for s, ms in zip(img.shape, min_shape)):
            filtered_images.append(img_path)
            filtered_masks.append(mask_path)
        else:
            print(f"❌ Volume trop petit : {os.path.basename(img_path)} shape={img.shape}, requis min {min_shape}")
    return filtered_images, filtered_masks

# --------- Main --------------
# def main(data_dir, epochs=20, batch_size=1, lr=1e-4):
# # def main(data_dir, epochs=2, batch_size=1, lr=1e-4):  # epochs réduits pour test rapide
#     images, masks = get_file_lists(data_dir)
#     # Prendre seulement 2 images pour valider la pipeline
#     # images, masks = images[:2], masks[:2]
#     train_imgs, val_imgs, train_masks, val_masks = train_test_split(
#         images, masks, test_size=0.2, random_state=42
#     )
#     train_dataset = Medical3DDataset(train_imgs, train_masks)
#     val_dataset = Medical3DDataset(val_imgs, val_masks)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


#     model = UNet3D().to(device)
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     best_dice = 0.0
#     os.makedirs('checkpoints', exist_ok=True)

#     for epoch in range(epochs):
#         train_loss = train_epoch(model, train_loader, criterion, optimizer)
#         val_loss, val_dice = val_epoch(model, val_loader, criterion)

#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

#         # Save best model
#         if val_dice > best_dice:
#             best_dice = val_dice
#             torch.save(model.state_dict(), f'checkpoints/best_model.pth')
#             print(f"New best model saved with Dice {best_dice:.4f}")

#         # Save checkpoint every 5 epochs
#         if (epoch + 1) % 5 == 0:
#             torch.save(model.state_dict(), f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
#             print(f"Checkpoint saved at epoch {epoch+1}")

#     # Final evaluation + save probability maps thresholded to masks
#     model.load_state_dict(torch.load('checkpoints/best_model.pth'))
#     model.eval()

#     os.makedirs('predictions', exist_ok=True)
#     with torch.no_grad():
#         for i, (img, mask) in enumerate(val_dataset):
#             img = img.unsqueeze(0).to(device)
#             pred = model(img).squeeze().cpu()
#             bin_pred = (pred > 0.5).float()

#             # Save binarized prediction as nifti
#             pred_np = bin_pred.permute(1,2,0).numpy()  # D,H,W -> H,W,D to match original? Attention au permute
#             # Ici on remet en (512,512,D)
#             pred_np = np.transpose(pred_np, (1,2,0))

#             affine = nib.load(val_imgs[i]).affine
#             nib.save(nib.Nifti1Image(pred_np, affine), f'predictions/pred_{i}.nii.gz')

#     print("Training and evaluation done.")

def main(data_dir, epochs=20, batch_size=1, lr=1e-4):
    images, masks = get_file_lists(data_dir)
    # Filtrer les volumes trop petits
    images, masks = filter_small_volumes(images, masks, min_shape=(8, 64, 64))
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    train_dataset = Medical3DDataset(train_imgs, train_masks)
    val_dataset = Medical3DDataset(val_imgs, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet3D().to(device)
    # model = UNet3D(in_channels=1, out_channels=1, features=[16, 32])  # 2 niveaux de pooling
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_dice = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice = val_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), f'checkpoints/best_model.pth')
            print(f"New best model saved with Dice {best_dice:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}")

    # Final evaluation + save probability maps thresholded to masks
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    save_challenge_predictions(model, val_dataset, val_imgs, device, save_dir="submission_labels", threshold=0.5)
    print("Training and evaluation done.")

if __name__ == "__main__":
    data_dir = "./dataset"  # dossier contenant fichiers .nii.gz
    main(data_dir)
