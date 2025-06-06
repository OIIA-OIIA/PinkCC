
# 📘 README – Détection de tumeurs ovariennes avec U-Net

## 🎯 Objectif du projet

L’objectif de ce projet est de développer un modèle d'intelligence artificielle capable de **segmenter les tumeurs ovariennes** à partir d'images médicales au format `.nii.gz` (IRM ou scanner).  
Nous utilisons une approche de **segmentation d’image** supervisée avec un modèle **U-Net**, très utilisé en imagerie médicale.

## 🧰 Structure des données

Nous disposons de deux dossiers principaux :
- `/images` : contient les images médicales brutes.
- `/labels` : contient les masques de segmentation (mêmes noms que les images), indiquant la présence de tumeurs.

Chaque image est en 3D (fichier `.nii.gz`), mais dans cette version du projet, nous travaillons avec des **coupes 2D** (par exemple, la 50ème coupe dans chaque volume).

## 🔧 Étapes du pipeline actuel

### 1. 📂 Chargement des données
- Utilisation de `nibabel` pour lire les fichiers `.nii.gz`.
- Extraction d'une coupe 2D spécifique pour chaque image et son masque.

### 2. 🧱 Dataset personnalisé
- Dataset PyTorch pour charger les images et masques 2D.
- Transformations : resize, tensor conversion, normalisation.
- Intégration avec le `DataLoader`.

### 3. 🧠 Modèle U-Net
- Encodeur → Bottleneck → Décodeur avec **skip connections**.
- Sortie : carte de probabilité des tumeurs.

### 4. 🎯 Fonction de perte : Dice Loss
- Utilisation du Dice coefficient pour gérer les classes déséquilibrées.
- Fonction de perte adaptée à la segmentation médicale.

### 5. 🏋️ Entraînement
- Optimiseur : Adam
- Batch size : 2
- Entraînement rapide pour valider le pipeline.

### 6. 👁️ Visualisation
- Affichage de l’image d’entrée, du masque réel et de la prédiction.

## 🚀 Ce qui fonctionne déjà

✅ Lecture et affichage des données  
✅ Préparation des données (Dataset + DataLoader)  
✅ Implémentation d’un U-Net fonctionnel  
✅ Entraînement de test du modèle  
✅ Visualisation des prédictions

## 📈 Prochaines étapes

- Ajouter une **fonction de validation** (Dice score, précision, rappel).
- Entraîner le modèle sur **plus d’images** et **plus d’epochs**, sur serveur GPU.
- Améliorer les performances :
  - 🔧 U-Net plus profond
  - 🧪 Data augmentation
  - 📐 Gestion des volumes 3D complets

## 👥 Équipe

Ce projet est développé dans le cadre d’un travail de groupe.


OIIA OIIA 
