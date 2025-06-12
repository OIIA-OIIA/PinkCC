#!/usr/bin/env python
# coding: utf-8

# # Génération de prédictions pour tous les fichiers d'images

import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé: {device}")

# Chemin vers le modèle pré-entraîné
MODEL_PATH = "best_model_gridsearch_20250606_223151.pth"  # Modifiez si nécessaire

# ## Modèle U-Net (copie depuis train_from_checkpoint.py)

class DoubleConv(nn.Module):
    """Deux convolutions + ReLU + BatchNorm"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(64, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)  # binaire

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        b = self.bottleneck(self.pool2(d2))

        u1 = self.up1(b)
        u1 = self.upconv1(torch.cat([u1, d2], dim=1))
        u2 = self.up2(u1)
        u2 = self.upconv2(torch.cat([u2, d1], dim=1))

        out = self.final_conv(u2)
        return torch.sigmoid(out)  # probas entre 0 et 1

# ## Dataset pour prédiction uniquement (sans labels)

class InferenceDataset(Dataset):
    def __init__(self, images_dir, slice_axis=2):
        self.images_dir = images_dir
        self.filenames = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
        self.slice_axis = slice_axis
        print(f"📁 Trouvé {len(self.filenames)} fichiers d'images")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.filenames[idx])
        filename = self.filenames[idx]
        
        # Charger l'image 3D
        image = nib.load(img_path).get_fdata()
        
        # Obtenir les dimensions originales pour reconstruction
        original_shape = image.shape
        
        # Extraire toutes les slices selon l'axe spécifié
        slices = []
        if self.slice_axis == 0:
            for i in range(image.shape[0]):
                slice_2d = image[i, :, :]
                slices.append(slice_2d)
        elif self.slice_axis == 1:
            for i in range(image.shape[1]):
                slice_2d = image[:, i, :]
                slices.append(slice_2d)
        else:  # slice_axis == 2
            for i in range(image.shape[2]):
                slice_2d = image[:, :, i]
                slices.append(slice_2d)
        
        # Normaliser et convertir en tenseurs
        processed_slices = []
        for slice_2d in slices:
            # Normalisation
            if np.max(slice_2d) > np.min(slice_2d):
                slice_normalized = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))
            else:
                slice_normalized = slice_2d
            
            # Convertir en tensor [1, H, W]
            slice_tensor = torch.tensor(slice_normalized, dtype=torch.float32).unsqueeze(0)
            processed_slices.append(slice_tensor)
        
        return processed_slices, filename, original_shape

# ## Fonction pour charger le modèle

def load_pretrained_model(model_path, device):
    """Charge un modèle pré-entraîné depuis un fichier .pth"""
    if not os.path.exists(model_path):
        print(f"❌ Erreur: Le fichier {model_path} n'existe pas!")
        return None
    
    try:
        model = UNet().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()  # Mode évaluation
        print(f"✅ Modèle chargé avec succès depuis: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# ## Fonction de calcul du Dice Score

def calculate_dice_score(pred, target, smooth=1e-6):
    """Calcule le score Dice entre prédiction et vérité terrain"""
    pred = (pred > 0.5).astype(np.float32)
    target = target.astype(np.float32)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# ## Fonction principale de génération des prédictions

def generate_all_predictions(images_dir, output_dir, model_path, device, threshold=0.5, labels_dir=None):
    """
    Génère des prédictions pour tous les fichiers du dossier images
    et les sauvegarde dans le dossier de sortie avec les mêmes noms
    Si labels_dir est fourni, calcule aussi le score Dice de validation
    """
    
    print(f"🚀 GÉNÉRATION DES PRÉDICTIONS")
    print(f"Dossier d'images: {images_dir}")
    print(f"Dossier de sortie: {output_dir}")
    print(f"Modèle: {model_path}")
    if labels_dir:
        print(f"Dossier de validation: {labels_dir}")
    print("-" * 60)
    
    # Vérifier que le dossier d'images existe
    if not os.path.exists(images_dir):
        print(f"❌ Le dossier d'images {images_dir} n'existe pas!")
        return False
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Dossier de sortie créé: {output_dir}")
    
    # Charger le modèle
    model = load_pretrained_model(model_path, device)
    if model is None:
        return False
    
    # Créer le dataset
    dataset = InferenceDataset(images_dir)
    if len(dataset) == 0:
        print("❌ Aucun fichier .nii.gz trouvé dans le dossier d'images!")
        return False
      # Traiter chaque volume
    print(f"\n🔄 Traitement de {len(dataset)} volumes...")
    
    dice_scores = []
    validation_available = labels_dir is not None and os.path.exists(labels_dir)
    
    if validation_available:
        print("📊 Calcul du score Dice de validation activé")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Génération des prédictions"):
            slices, filename, original_shape = dataset[idx]
            
            # Charger le label de vérité terrain si disponible
            ground_truth = None
            if validation_available:
                label_path = os.path.join(labels_dir, filename)
                if os.path.exists(label_path):
                    try:
                        ground_truth = nib.load(label_path).get_fdata()
                    except Exception as e:
                        print(f"⚠️ Erreur chargement label {filename}: {e}")
            
            # Traiter toutes les slices du volume
            predicted_slices = []
            
            for slice_tensor in slices:
                # Ajouter une dimension batch [1, 1, H, W]
                slice_batch = slice_tensor.unsqueeze(0).to(device)
                
                # Prédiction
                pred = model(slice_batch)
                
                # Appliquer le seuil et convertir en uint8
                pred_binary = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)
                predicted_slices.append(pred_binary)
            
            # Reconstruire le volume 3D selon l'axe original
            if dataset.slice_axis == 0:
                prediction_volume = np.stack(predicted_slices, axis=0)
            elif dataset.slice_axis == 1:
                prediction_volume = np.stack(predicted_slices, axis=1)
            else:  # slice_axis == 2
                prediction_volume = np.stack(predicted_slices, axis=2)
            
            # Calculer le score Dice si ground truth disponible
            dice_score = None
            if ground_truth is not None:
                try:
                    dice_score = calculate_dice_score(prediction_volume, ground_truth)
                    dice_scores.append(dice_score)
                except Exception as e:
                    print(f"⚠️ Erreur calcul Dice pour {filename}: {e}")
            
            # Vérifier que les dimensions correspondent
            if prediction_volume.shape != original_shape:
                print(f"⚠️ Attention: Dimensions différentes pour {filename}")
                print(f"   Original: {original_shape}, Prédiction: {prediction_volume.shape}")
            
            # Sauvegarder en format .nii.gz
            nii_img = nib.Nifti1Image(prediction_volume, affine=np.eye(4))
            output_path = os.path.join(output_dir, filename)
            nib.save(nii_img, output_path)
            
            # Afficher le résultat avec Dice si disponible
            if dice_score is not None:
                print(f"✅ Sauvegardé: {filename} - Shape: {prediction_volume.shape} - Dice: {dice_score:.4f}")
            else:
                print(f"✅ Sauvegardé: {filename} - Shape: {prediction_volume.shape}")
    
    print(f"\n🎉 GÉNÉRATION TERMINÉE!")
    print(f"📁 {len(dataset)} prédictions sauvegardées dans: {output_dir}")
    
    # Afficher les statistiques Dice si disponibles
    if dice_scores:
        avg_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        min_dice = np.min(dice_scores)
        max_dice = np.max(dice_scores)
        
        print(f"\n📊 STATISTIQUES DE VALIDATION:")
        print(f"Nombre de volumes validés: {len(dice_scores)}")
        print(f"Score Dice moyen: {avg_dice:.4f} (±{std_dice:.4f})")
        print(f"Score Dice min: {min_dice:.4f}")
        print(f"Score Dice max: {max_dice:.4f}")
        
        # Classification de performance
        if avg_dice >= 0.8:
            performance = "🟢 EXCELLENTE"
        elif avg_dice >= 0.7:
            performance = "🟡 BONNE"
        elif avg_dice >= 0.6:
            performance = "🟠 MOYENNE"
        else:
            performance = "🔴 FAIBLE"
        
        print(f"Performance globale: {performance}")
    
    return True

# ## Fonction pour vérifier la structure des dossiers

def setup_directories():
    """Vérifie et crée la structure de dossiers nécessaire"""
    
    # Chemins des dossiers
    dataset_dir = "./final_dataset"
    images_dir = "./final_dataset/images"
    labels_dir = "./final_dataset/labels"
    validation_labels_dir = "./final_dataset/labels"  # Labels de validation (existants)
    
    print("🔍 Vérification de la structure des dossiers...")
    
    # Vérifier si le dossier DatasetChallenge existe
    if not os.path.exists(dataset_dir):
        print(f"❌ Le dossier {dataset_dir} n'existe pas!")
        print("📦 Peut-être devez-vous extraire les fichiers ZIP d'abord?")
        
        # Lister les fichiers ZIP disponibles
        zip_files = [f for f in os.listdir(".") if f.endswith(".zip")]
        if zip_files:
            print("📁 Fichiers ZIP trouvés:")
            for zip_file in zip_files:
                print(f"  - {zip_file}")
            print("💡 Conseil: Utilisez unzip_script.py pour extraire les données")
        return None, None, None
    
    # Vérifier le dossier images
    if not os.path.exists(images_dir):
        print(f"❌ Le dossier {images_dir} n'existe pas!")
        return None, None, None
    
    # Créer un dossier séparé pour les nouvelles prédictions
    new_predictions_dir = "./final_dataset/predictions_generated"
    if not os.path.exists(new_predictions_dir):
        print(f"📁 Création du dossier {new_predictions_dir}...")
        os.makedirs(new_predictions_dir, exist_ok=True)
    
    # Compter les fichiers
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz')]
    print(f"✅ Dossier images: {len(image_files)} fichiers .nii.gz")
    
    # Vérifier si les labels de validation existent
    if os.path.exists(validation_labels_dir):
        label_files = [f for f in os.listdir(validation_labels_dir) if f.endswith('.nii.gz')]
        print(f"✅ Labels de validation: {len(label_files)} fichiers .nii.gz")
    else:
        validation_labels_dir = None
        print("⚠️ Pas de labels de validation trouvés")
    
    return images_dir, new_predictions_dir, validation_labels_dir

# ## SCRIPT PRINCIPAL

if __name__ == "__main__":
    print("🏥 GÉNÉRATEUR DE PRÉDICTIONS POUR SEGMENTATION MÉDICALE")
    print("=" * 60)
    
    # Vérifier et configurer les dossiers
    images_dir, predictions_dir, validation_labels_dir = setup_directories()
    
    if images_dir is None:
        print("❌ Impossible de continuer sans le dossier d'images!")
        print("💡 Assurez-vous que ./final_dataset/images existe et contient des fichiers .nii.gz")
        exit(1)
    
    # Générer les prédictions avec validation si possible
    success = generate_all_predictions(
        images_dir=images_dir,
        output_dir=predictions_dir,
        model_path=MODEL_PATH,
        device=device,
        threshold=0.5,
        labels_dir=validation_labels_dir
    )
    
    if success:
        print(f"\n✅ SUCCÈS! Toutes les prédictions ont été générées")
        print(f"📁 Vérifiez le dossier: {predictions_dir}")
        
        # Lister les fichiers créés
        created_files = [f for f in os.listdir(predictions_dir) if f.endswith('.nii.gz')]
        print(f"📝 {len(created_files)} fichiers de labels créés:")
        for i, filename in enumerate(sorted(created_files)[:10]):  # Afficher les 10 premiers
            print(f"  {i+1:2d}. {filename}")
        if len(created_files) > 10:
            print(f"     ... et {len(created_files) - 10} autres fichiers")
        
        if validation_labels_dir:
            print(f"\n💡 Les scores Dice de validation ont été calculés et affichés ci-dessus")
        else:
            print(f"\n💡 Aucune validation effectuée (pas de labels de référence trouvés)")
    else:
        print("❌ Échec de la génération des prédictions!")
