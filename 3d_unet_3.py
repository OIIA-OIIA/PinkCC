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

# --------- Dataset 3D (CORRIGÉ POUR MULTICLASS) -------------
class Medical3DDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, debug=False, target_class=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.debug = debug
        self.target_class = target_class  # Si None, combine toutes les classes > 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        if self.debug:
            print(f"📊 Image {idx}: shape={img.shape}, min={img.min():.3f}, max={img.max():.3f}")
            print(f"📊 Mask {idx}: shape={mask.shape}, min={mask.min():.3f}, max={mask.max():.3f}, unique={np.unique(mask)}")
            unique_vals, counts = np.unique(mask, return_counts=True)
            total_pixels = mask.size
            for val, count in zip(unique_vals, counts):
                percentage = (count / total_pixels) * 100
                print(f"   Classe {val}: {count} pixels ({percentage:.2f}%)")

        # Normalisation image robuste (percentile clipping)
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = np.zeros_like(img)

        # Gestion des masques multiclasses
        if self.target_class is not None:
            # Segmentation d'une classe spécifique
            mask = (mask == self.target_class).astype(np.float32)
        else:
            # Segmentation binaire: toutes les classes > 0 vs background
            mask = (mask > 0).astype(np.float32)

        # Vérification du déséquilibre
        positive_ratio = mask.sum() / mask.size
        if self.debug:
            print(f"🎯 Ratio pixels positifs: {positive_ratio:.6f} ({positive_ratio*100:.4f}%)")

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

# --------- Loss pour données déséquilibrées (NOUVEAU) -------------
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight

    def focal_loss(self, preds, targets):
        bce_loss = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

    def dice_loss(self, preds, targets):
        return 1 - dice_score(preds, targets)

    def forward(self, preds, targets):
        focal = self.focal_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        return (1 - self.dice_weight) * focal + self.dice_weight * dice

# --------- Loss avec poids automatiques (ALTERNATIVE) -------------
class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        # Calcul automatique du poids pour BCE
        pos_weight = (targets == 0).sum().float() / (targets == 1).sum().float()
        pos_weight = torch.clamp(pos_weight, min=1.0, max=100.0)  # Limiter le poids

        bce_loss = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
        bce_loss = bce_loss * (targets * pos_weight + (1 - targets))
        bce_loss = bce_loss.mean()

        dice_loss = 1 - dice_score(preds, targets)
        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss

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

# --------- Test minimal (CORRIGÉ POUR DÉSÉQUILIBRE) ----------
def test_minimal_dataset(data_dir, n_samples=2, target_class=None):
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
    if target_class is not None:
        print(f"🎯 Classe cible: {target_class}")
    else:
        print("🎯 Mode binaire: toutes classes > 0")

    # Dataset avec debug
    dataset = Medical3DDataset(test_images, test_masks, debug=True, target_class=target_class)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Analyser le déséquilibre des données
    total_positive = 0
    total_pixels = 0
    for img, mask in dataset:
        total_positive += mask.sum().item()
        total_pixels += mask.numel()

    positive_ratio = total_positive / total_pixels
    print(f"📊 Déséquilibre global: {positive_ratio:.6f} ({positive_ratio*100:.4f}% pixels positifs)")

    if positive_ratio < 1e-5:
        print("❌ Données trop déséquilibrées (< 0.001% pixels positifs)")
        print("💡 Suggestions:")
        print("   - Essayez avec une classe spécifique: target_class=1 ou 2")
        print("   - Vérifiez que vos masques contiennent bien des annotations")
        return False

    if positive_ratio < 1e-3:
        print("⚠️ Données très déséquilibrées, utilisation de Focal Loss")
        criterion = FocalDiceLoss(alpha=0.75, gamma=2.0)
    else:
        print("✅ Déséquilibre gérable, utilisation de BCE pondérée")
        criterion = WeightedBCEDiceLoss()

    # Modèle très simple
    model = UNet3D(features=[8, 16]).to(device)  # Très petit pour test rapide
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # LR plus élevé

    print("\n🏋️ Entraînement sur échantillons minimaux...")

    # Plus d'epochs pour les données déséquilibrées
    max_epochs = 10 if positive_ratio < 1e-4 else 5

    best_dice = 0.0
    for epoch in range(max_epochs):
        train_loss, train_dice = train_epoch(model, loader, criterion, optimizer, debug=(epoch==0))
        print(f"Mini-Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Dice={train_dice:.4f}")

        if train_dice > best_dice:
            best_dice = train_dice

        # Critères d'arrêt adaptés au déséquilibre
        if positive_ratio < 1e-4:  # Très déséquilibré
            if train_dice > 0.01:
                print("✅ Le modèle commence à apprendre sur données très déséquilibrées")
                return True
        else:  # Modérément déséquilibré
            if train_dice > 0.1:
                print("✅ Le modèle semble apprendre correctement")
                return True

    print(f"📈 Meilleur Dice obtenu: {best_dice:.4f}")

    if positive_ratio < 1e-4 and best_dice > 0.005:
        print("⚠️ Apprentissage très lent mais détectable (données extrêmement déséquilibrées)")
        return True
    elif best_dice > 0.05:
        print("⚠️ Apprentissage lent mais possible")
        return True
    else:
        print("❌ Aucun apprentissage détecté")
        print("💡 Suggestions supplémentaires:")
        print("   - Vérifiez l'alignement spatial images/masques")
        print("   - Testez avec un sous-ensemble des données (crop des régions d'intérêt)")
        print("   - Vérifiez les unités des masques (Hounsfield, mm, etc.)")
        return False

# --------- Version complète (CORRIGÉE) ----------
def main_full(data_dir, epochs=20, batch_size=1, lr=1e-4, target_class=None):
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
    if target_class is not None:
        print(f"🎯 Classe cible: {target_class}")

    train_dataset = Medical3DDataset(train_imgs, train_masks, target_class=target_class)
    val_dataset = Medical3DDataset(val_imgs, val_masks, target_class=target_class)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Analyser le déséquilibre pour choisir la loss
    sample_img, sample_mask = train_dataset[0]
    positive_ratio = sample_mask.sum().item() / sample_mask.numel()

    # if positive_ratio < 1e-4:
    #     print(f"📊 Données très déséquilibrées ({positive_ratio:.6f}), utilisation Focal Loss")
    #     criterion = FocalDiceLoss(alpha=0.75, gamma=2.0)
    # else:
    #     print(f"📊 Déséquilibre modéré ({positive_ratio:.6f}), utilisation BCE pondérée")
    #     criterion = WeightedBCEDiceLoss()

    criterion = FocalDiceLoss(alpha=0.75, gamma=2.0)

    model = UNet3D(features=[16, 32, 64]).to(device)  # Taille raisonnable
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
                'target_class': target_class,
            }, 'checkpoints/best_model.pth')
            print(f"💾 Nouveau meilleur modèle: Dice={best_dice:.4f}")

        # Early stopping adapté au déséquilibre
        min_dice_threshold = 0.001 if positive_ratio < 1e-4 else 0.01
        if epoch > 10 and val_dice < min_dice_threshold:
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

    # # Options pour gérer le multiclass
    # print("🎯 Options de segmentation:")
    # print("  0: Binaire (toutes classes > 0 vs background)")
    # print("  1: Classe 1 seulement")
    # print("  2: Classe 2 seulement")

    # try:
    #     choice = input("Choisissez une option (0/1/2) [défaut: 0]: ").strip()
    #     if choice == "1":
    #         target_class = 1
    #     elif choice == "2":
    #         target_class = 2
    #     else:
    #         target_class = None
    # except:
    #     target_class = None

    # # 1. D'abord test minimal
    # print("PHASE 1: Test minimal")
    # success = test_minimal_dataset(data_dir, n_samples=2, target_class=target_class)

    # if success:
    #     print("\n" + "="*50)
    #     print("PHASE 2: Entraînement complet")
    #     response = input("Le test minimal a réussi. Lancer l'entraînement complet? (y/n): ")
    #     if response.lower() == 'y':
    #         main_full(data_dir, epochs=30, batch_size=1, lr=1e-4, target_class=target_class)
    # else:
    #     print("\n❌ Le test minimal a échoué.")
    #     if target_class is None:
    #         print("💡 Essayez avec une classe spécifique (relancez le script et choisissez 1 ou 2)")
    #     else:
    #         print("💡 Vérifiez vos données et leur alignement spatial")
    print("PHASE 2: Entraînement complet")
    main_full(data_dir, epochs=30, batch_size=1, lr=1e-4, target_class=None)
