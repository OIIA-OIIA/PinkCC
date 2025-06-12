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
print(f"🚀 Using device: {device}")

# --------- Dataset 3D (CORRIGÉ) -------------
class Medical3DDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, debug=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.debug = debug

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        if self.debug:
            print(f"📊 Image {idx}: shape={img.shape}, min={img.min():.3f}, max={img.max():.3f}")
            print(f"📊 Mask {idx}: shape={mask.shape}, min={mask.min():.3f}, max={mask.max():.3f}, unique={np.unique(mask)}")

        # Normalisation image (avec gestion des cas particuliers)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = np.zeros_like(img)

        # Binarisation du masque (CORRIGÉ)
        mask = (mask > 0.5).astype(np.float32)  # Seuil plus strict

        # CORRECTION: Dimensions (H,W,D) -> (D,H,W) pour les convolutions 3D
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)  # (H,W,D) -> (D,H,W)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2,0,1)  # (H,W,D) -> (D,H,W)

        # Ajout dimension channel
        img = img.unsqueeze(0)   # (1,D,H,W)
        mask = mask.unsqueeze(0) # (1,D,H,W)

        if self.debug:
            print(f"🔄 Final tensor shapes - img: {img.shape}, mask: {mask.shape}")
            print(f"🔄 Mask values after processing: {torch.unique(mask)}")

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

# --------- 3D U-Net (SIMPLIFIÉ POUR DEBUG) --------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64]):  # Plus petit pour debug
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

        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = nn.functional.max_pool3d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsampling
            skip = skip_connections[idx//2]

            # CORRECTION: Gestion des tailles différentes
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)  # Conv block

        return torch.sigmoid(self.final_conv(x))

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),  # Ajout BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),  # Ajout BatchNorm
            nn.ReLU(inplace=True),
        )

# --------- Dice score (CORRIGÉ) -------------
def dice_score(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    targets = targets.float()

    # Calcul par batch puis moyenne
    batch_size = preds.shape[0]
    dice_scores = []

    for i in range(batch_size):
        pred_i = preds[i].flatten()
        target_i = targets[i].flatten()

        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()

        if union == 0:
            dice_i = 1.0 if pred_i.sum() == 0 else 0.0
        else:
            dice_i = (2. * intersection + smooth) / (union + smooth)

        dice_scores.append(dice_i)

    return torch.tensor(dice_scores).mean()

# --------- Loss combinée (NOUVEAU) -------------
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = 1 - dice_score(preds, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

# --------- Prepare data lists (CORRIGÉ) ----------
def get_file_lists(data_dir, debug=False):
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    masks = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])

    assert len(images) == len(masks), f"Mismatch: {len(images)} images vs {len(masks)} masks"

    if debug:
        print(f"📁 Found {len(images)} image/mask pairs")
        for i, (img_path, mask_path) in enumerate(zip(images[:3], masks[:3])):
            img = nib.load(img_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()
            print(f"   {i}: {os.path.basename(img_path)} -> shape {img.shape}")
            print(f"      mask unique values: {np.unique(mask)}")

    return images, masks

# --------- Filtre volumes (CORRIGÉ) ----------
def filter_small_volumes(images, masks, min_depth=8, min_hw=32):
    """
    Filtre les volumes avec une profondeur minimale et des dimensions H,W minimales
    Format attendu: (H, W, D)
    """
    filtered_images = []
    filtered_masks = []

    for img_path, mask_path in zip(images, masks):
        img = nib.load(img_path).get_fdata()
        h, w, d = img.shape

        if d >= min_depth and h >= min_hw and w >= min_hw:
            filtered_images.append(img_path)
            filtered_masks.append(mask_path)
        else:
            print(f"❌ Volume filtré : {os.path.basename(img_path)} shape={img.shape}, requis: D>={min_depth}, H,W>={min_hw}")

    print(f"✅ {len(filtered_images)}/{len(images)} volumes conservés après filtrage")
    return filtered_images, filtered_masks

# --------- Training function (AMÉLIORÉE) ----------
def train_epoch(model, loader, criterion, optimizer, debug=False):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    loop = tqdm(loader, desc='Training', leave=False)
    for batch_idx, (imgs, masks) in enumerate(loop):
        imgs, masks = imgs.to(device), masks.to(device)

        if debug and batch_idx == 0:
            print(f"🔍 Batch 0 - imgs: {imgs.shape}, masks: {masks.shape}")
            print(f"🔍 Mask range: {masks.min():.3f} to {masks.max():.3f}")

        optimizer.zero_grad()
        preds = model(imgs)

        if debug and batch_idx == 0:
            print(f"🔍 Predictions shape: {preds.shape}, range: {preds.min():.3f} to {preds.max():.3f}")

        loss = criterion(preds, masks)
        dice = dice_score(preds, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += dice.item()

        loop.set_postfix(loss=loss.item(), dice=dice.item())

    return running_loss / len(loader), running_dice / len(loader)

# --------- Validation function (AMÉLIORÉE) ----------
def val_epoch(model, loader, criterion, debug=False):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc='Validation', leave=False)
        for batch_idx, (imgs, masks) in enumerate(loop):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            loss = criterion(preds, masks)
            dice = dice_score(preds, masks)

            running_loss += loss.item()
            running_dice += dice.item()

            if debug and batch_idx == 0:
                print(f"🔍 Val batch 0 - Dice: {dice.item():.4f}")

            loop.set_postfix(loss=loss.item(), dice=dice.item())

    return running_loss / len(loader), running_dice / len(loader)

# --------- Test minimal (NOUVEAU) ----------
def test_minimal_dataset(data_dir, n_samples=2):
    """
    Test sur un dataset minimal pour vérifier que le modèle peut apprendre
    """
    print("🧪 TEST MINIMAL - Vérification de l'apprentissage")
    print("="*50)

    images, masks = get_file_lists(data_dir, debug=True)
    images, masks = filter_small_volumes(images, masks, min_depth=8, min_hw=32)

    if len(images) < n_samples:
        print(f"❌ Pas assez de volumes: {len(images)} < {n_samples}")
        return False

    # Prendre seulement les premiers échantillons
    test_images = images[:n_samples]
    test_masks = masks[:n_samples]

    print(f"🔬 Test avec {len(test_images)} échantillons")

    # Dataset avec debug
    dataset = Medical3DDataset(test_images, test_masks, debug=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Modèle très simple
    model = UNet3D(features=[8, 16]).to(device)  # Très petit pour test rapide
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n🏋️ Entraînement sur échantillons minimaux...")

    # Quelques epochs pour voir si ça apprend
    for epoch in range(5):
        train_loss, train_dice = train_epoch(model, loader, criterion, optimizer, debug=(epoch==0))
        print(f"Mini-Epoch {epoch+1}: Loss={train_loss:.4f}, Dice={train_dice:.4f}")

        # Si le dice commence à monter, c'est bon signe
        if train_dice > 0.1:
            print("✅ Le modèle semble apprendre (Dice > 0.1)")
            return True

    if train_dice < 0.01:
        print("❌ Le modèle n'apprend pas (Dice < 0.01)")
        return False
    else:
        print("⚠️ Apprentissage lent mais possible")
        return True

# --------- Version complète (CORRIGÉE) ----------
def main_full(data_dir, epochs=20, batch_size=1, lr=1e-4):
    print("🚀 ENTRAÎNEMENT COMPLET")
    print("="*50)

    images, masks = get_file_lists(data_dir)
    images, masks = filter_small_volumes(images, masks, min_depth=8, min_hw=32)

    if len(images) < 4:
        print("❌ Pas assez de données pour train/val split")
        return

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    print(f"📊 Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    train_dataset = Medical3DDataset(train_imgs, train_masks)
    val_dataset = Medical3DDataset(val_imgs, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet3D(features=[16, 32, 64]).to(device)  # Taille raisonnable
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_dice = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    # Historique pour plot
    train_losses, val_losses, val_dices = [], [], []

    for epoch in range(epochs):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice = val_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train: L={train_loss:.4f} D={train_dice:.4f} | "
              f"Val: L={val_loss:.4f} D={val_dice:.4f} | "
              f"LR={current_lr:.2e}")

        # Historique
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        # Sauvegarde du meilleur modèle
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, 'checkpoints/best_model.pth')
            print(f"💾 Nouveau meilleur modèle: Dice={best_dice:.4f}")

        # Early stopping si pas d'amélioration
        if epoch > 10 and val_dice < 0.01:
            print("⚠️ Early stopping: pas d'apprentissage détecté")
            break

    # Plot des résultats
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Losses')

    plt.subplot(1, 2, 2)
    plt.plot(val_dices, label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Dice Score')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    print(f"✅ Entraînement terminé. Meilleur Dice: {best_dice:.4f}")

if __name__ == "__main__":
    data_dir = "./dataset"

    # 1. D'abord test minimal
    print("PHASE 1: Test minimal")
    success = test_minimal_dataset(data_dir, n_samples=2)

    if success:
        print("\n" + "="*50)
        print("PHASE 2: Entraînement complet")
        response = input("Le test minimal a réussi. Lancer l'entraînement complet? (y/n): ")
        if response.lower() == 'y':
            main_full(data_dir, epochs=30, batch_size=1, lr=1e-4)
    else:
        print("\n❌ Le test minimal a échoué. Vérifiez vos données avant de continuer.")
        print("Suggestions:")
        print("- Vérifiez que les masques ne sont pas vides")
        print("- Vérifiez la correspondance images/masques")
        print("- Vérifiez les valeurs dans les masques (0/1 ou 0/255?)")
