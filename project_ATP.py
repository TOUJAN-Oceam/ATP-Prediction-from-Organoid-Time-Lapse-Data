import os
import glob
import re
import math
import random
from collections import defaultdict
from torchvision.transforms.v2 import Normalize # <- À ajouter en haut avec vos imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.v2 as transforms 
import torchvision.models as models

import time
import matplotlib.pyplot as plt


# === À DÉFINIR AU DÉBUT DU SCRIPT ===
train_dir = "DATASET/train"  # Remplacez par votre vrai chemin si besoin

# Le pattern que vous aviez créé est parfait pour extraire l'ATP et le puits
WELL_DIR_PATTERN_TRAIN = re.compile(
    r"^(?P<well_id>[0-9a-fA-F]+)_(?P<atp>\d+)_(?P<conc>[A-Za-z]+)_(?P<date>\d+)$"
)

# ==========================================
# NOUVEAU : Compteur global de fichiers corrompus
# ==========================================
CORRUPTION_STATS = {
    "total_attempts": 0,
    "corrupted_files": 0
}

# ==========================================
# 0. VÉRIFICATION DU GPU (À ajouter ici)
# ==========================================
print("="*50)
print(f"Version de PyTorch : {torch.__version__}")
if torch.cuda.is_available():
    print(f"✅ GPU détecté : {torch.cuda.get_device_name(0)}")
    print(f"✅ Nombre de GPUs : {torch.cuda.device_count()}")
    # Optionnel : Vider le cache mémoire du GPU avant de commencer
    torch.cuda.empty_cache() 
else:
    print("❌ ATTENTION : Aucun GPU CUDA détecté.")
    print("L'apprentissage se fera sur le CPU, ce qui sera très lent.")
    print("Vérifiez l'installation de PyTorch avec le support CUDA.")
print("="*50)




def plot_learning_curves(history, patient_id):
    """Génère et sauvegarde un graphique des courbes d'apprentissage."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- Axe de gauche : MAE Log (Train vs Val) ---
    color = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('MAE (Log)', color=color, fontsize=12)
    ax1.plot(epochs, history['train_loss'], color='darkblue', label='Train MAE (Log)', marker='o', linewidth=2)
    ax1.plot(epochs, history['val_loss'], color='deepskyblue', label='Val MAE (Log)', marker='s', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # --- Axe de droite : MAPE (%) ---
    ax2 = ax1.twinx()  # Instancie un second axe Y qui partage le même axe X
    color = 'tab:red'
    ax2.set_ylabel('Val MAPE (%)', color=color, fontsize=12)
    ax2.plot(epochs, history['val_mape'], color='crimson', label='Val MAPE (%)', marker='^', linestyle='dashed', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    plt.title(f'Courbes d\'apprentissage - Patient en validation : {patient_id}', fontsize=14)
    fig.tight_layout()  
    
    # Sauvegarde de l'image
    plt.savefig(f'learning_curve_{patient_id}.png', dpi=300)
    plt.close()
    print(f"    📊 Graphique sauvegardé sous : learning_curve_{patient_id}.png")



# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================

def get_well_directories(data_dir: str, pattern: re.Pattern):
    """(Votre fonction existante) Récupère toutes les vidéos."""
    wells = []
    for patient_id in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_id)
        if not os.path.isdir(patient_path): continue
        for well_dir_name in os.listdir(patient_path):
            well_path = os.path.join(patient_path, well_dir_name)
            if not os.path.isdir(well_path): continue
            match = pattern.match(well_dir_name)
            if match:
                tif_files = glob.glob(os.path.join(well_path, "*.tif"))
                for tif_file in tif_files:
                    well = {
                        "patient_id": patient_id,
                        "well_id": match.group("well_id"),
                        "tif_path": os.path.abspath(tif_file),
                    }
                    if "atp" in match.groupdict():
                        well["atp"] = int(match.group("atp"))
                    wells.append(well)
    return wells

def group_by_well(cavity_list):
    """Regroupe les cavités individuelles par identifiant de puits."""
    wells_data = defaultdict(lambda: {"cavities": [], "atp": None, "patient_id": None})
    for item in cavity_list:
        wid = item["well_id"]
        wells_data[wid]["cavities"].append(item["tif_path"])
        if "atp" in item:
            wells_data[wid]["atp"] = item["atp"]
        wells_data[wid]["patient_id"] = item["patient_id"]
    return list(wells_data.items())

import torchvision.transforms.v2 as transforms

class WellDatasetMIL(Dataset):
    def __init__(self, wells_list, img_size=224, is_train=True, sample_frac=0.50):
        self.wells_list = wells_list
        self.img_size = img_size
        self.is_train = is_train
        self.sample_frac = sample_frac
        
        # Normalisation standard RGB
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

        # Augmentation de données pour limiter l'overfit
        if self.is_train:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
            ])
        else:
            self.augment = None

    def __len__(self):
        return len(self.wells_list)

    def __getitem__(self, idx):
        well_id, well_data = self.wells_list[idx]
        cavity_paths = well_data["cavities"]
        
        if self.is_train:
            num_to_sample = max(1, int(len(cavity_paths) * self.sample_frac))
            sampled_paths = random.sample(cavity_paths, num_to_sample)
        else:
            sampled_paths = cavity_paths
            
        frame_indices = [0, 2, 5, 7] 
        all_cavities_tensors = []
        
        for path in sampled_paths:
            # =======================================================
            # NOUVEAU : Bloc Try/Except pour gérer les images corrompues
            # =======================================================
            try:
                img = Image.open(path)
                # Astuce : PIL charge l'image de manière "paresseuse". 
                # On force img.load() pour déclencher l'erreur immédiatement si le fichier est tronqué.
                img.load() 
                
                frames = []
                for i in frame_indices:
                    img.seek(i if i < img.n_frames else img.n_frames - 1)
                    frame = transforms.functional.to_image(img).float() / 255.0
                    frame = frame.repeat(3, 1, 1) # Pseudo-RGB
                    frames.append(frame)
                
                cavity_tensor = torch.stack(frames, dim=0) 
                
                if self.augment is not None:
                    cavity_tensor = self.augment(cavity_tensor)
                    
                cavity_tensor = self.normalize(cavity_tensor)
                cavity_tensor = transforms.functional.resize(cavity_tensor, (self.img_size, self.img_size))
                
                all_cavities_tensors.append(cavity_tensor)
                
            except Exception as e:
                # Si l'image plante, on affiche le problème et on l'ignore !
                print(f"\n⚠️ Fichier ignoré (corrompu ou illisible) : {path}")
                print(f"   Détail : {e}")
                continue # Passe à l'image suivante de la boucle for
            # =======================================================

        # =======================================================
        # SÉCURITÉ : Que faire si TOUTES les images d'un puits sont corrompues ?
        # =======================================================
        if len(all_cavities_tensors) == 0:
            print(f"\n❌ ALERTE : Le puits {well_id} n'a aucune image valide ! Génération d'un tenseur vide de secours.")
            # On crée une fausse "cavité noire" (1, 4 frames, 3 canaux, 224, 224) pour éviter le crash de PyTorch
            dummy_tensor = torch.zeros((1, 4, 3, self.img_size, self.img_size))
            all_cavities_tensors.append(dummy_tensor)

        well_tensor = torch.stack(all_cavities_tensors, dim=0)

        # Retourne les infos complètes pour les prédictions
        if well_data["atp"] is not None:
            y = torch.tensor([math.log1p(well_data["atp"])], dtype=torch.float32)
            return well_tensor, y, well_id, well_data["patient_id"]
        else:
            return well_tensor, well_id, well_data["patient_id"]
        

# ==========================================
# 3. MODÈLE : ATTENTION-BASED MIL (AVEC RESNET18)
# ==========================================

class LSTM_MILNetwork(nn.Module):
    def __init__(self):
        super(LSTM_MILNetwork, self).__init__()
        
        # 1. Le CNN (L'oeil)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]) # Sort 512
        
        # On gèle les premières couches pour stopper l'overfitting massif
        for name, param in self.cnn.named_parameters():
            if "layer4" not in name and "layer3" not in name:
                param.requires_grad = False
                
        # 2. Le LSTM (La mémoire temporelle)
        # Il lit des vecteurs de 512, et résume la vidéo en un vecteur de 128
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        
        # 3. L'Attention (Décide de l'importance de chaque cavité)
        # Remplacer self.attention par :
        self.attention_V = nn.Sequential(nn.Linear(128, 64), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid())
        self.attention_weights = nn.Linear(64, 1)
        
        # 4. Le Régresseur (Prédit l'ATP global du puits)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # On garde un fort Dropout contre l'overfit
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (1_batch, N_cavités, 4_frames, 3_canaux, 224, 224)
        x = x.squeeze(0) 
        N, S, C, H, W = x.size() # N=Cavités, S=Frames (Sequence)
        
        # Etape A : On aplatit
        x_flat = x.view(N * S, C, H, W)
        
        # ==========================================
        # L'ASTUCE ANTI-CRASH : LE CHUNKING
        # ==========================================
        chunk_size = 32 # Taille de lot digeste pour le GPU (baissez à 16 si ça crash encore)
        features_list = []
        
        # On fait passer les images par paquets de 'chunk_size'
        for i in range(0, x_flat.size(0), chunk_size):
            chunk = x_flat[i : i + chunk_size]
            chunk_feat = self.cnn(chunk) # ResNet traite 32 images max
            features_list.append(chunk_feat)
            
        # On recolle tous les résultats mathématiques ensemble
        features = torch.cat(features_list, dim=0) # Shape: (N * S, 512, 1, 1)
        # ==========================================
        
        features = features.view(N, S, 512) # On redécoupe en séquences (N_cavités, 4_frames, 512)
        
        # Etape B : Analyse temporelle par le LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # On récupère "la dernière pensée" du LSTM
        cavity_embeddings = h_n[0] # Shape: (N_cavités, 128)
        
        # Etape C : Agrégation MIL avec Attention
        # Remplacer les lignes d'attention (Etape C) par :
        A_V = self.attention_V(cavity_embeddings)
        A_U = self.attention_U(cavity_embeddings)
        attn_scores = self.attention_weights(A_V * A_U) # Gated Mechanism
        attn_weights = torch.softmax(attn_scores, dim=0)
        
        well_embedding = torch.sum(cavity_embeddings * attn_weights, dim=0, keepdim=True) # (1, 128)
        
        # Etape D : Prédiction
        out = self.regressor(well_embedding)
        return out
    

# ==========================================
# 4. BOUCLE D'ENTRAÎNEMENT & VRAIE CV LOSO
# ==========================================
import time

def calculate_mape(y_true_raw, y_pred_raw):
    return torch.mean(torch.abs((y_true_raw - y_pred_raw) / y_true_raw)) * 100

def train_and_validate_loso(grouped_wells_list, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] Démarrage sur : {device}")

    patients = list(set([data["patient_id"] for _, data in grouped_wells_list]))
    print(f"Nombre total de patients (Folds) : {len(patients)}")
    print("-" * 80)

    fold_mapes = []

    for fold_idx, val_patient in enumerate(patients):
        print(f"\n[{time.strftime('%H:%M:%S')}] === FOLD {fold_idx + 1}/{len(patients)} | Patient en Validation : {val_patient} ===")
        
        train_data = [item for item in grouped_wells_list if item[1]["patient_id"] != val_patient]
        val_data = [item for item in grouped_wells_list if item[1]["patient_id"] == val_patient]

        train_dataset = WellDatasetMIL(train_data, is_train=True, sample_frac=0.25)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        val_dataset = WellDatasetMIL(val_data, is_train=False) 
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        model = LSTM_MILNetwork().to(device)
        criterion = nn.SmoothL1Loss(beta=1.0)
        # Astuce : On baisse légèrement le Learning Rate (de 1e-4 à 5e-5) pour réduire l'instabilité (le yo-yo)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)
        
        best_val_mape = float('inf')

        history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # --- ENTRAÎNEMENT ---
            model.train()
            train_loss_log = 0.0
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                predictions = model(x)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()
                train_loss_log += loss.item()

            # --- VALIDATION ---
            model.eval()
            val_loss_log = 0.0
            val_mape = 0.0
            val_mae_brute = 0.0
            
            with torch.no_grad():
                for x, y_log, _ in val_loader:
                    x, y_log = x.to(device), y_log.to(device)
                    pred_log = model(x)
                    
                    # 1. MAE Log (Objectif = 0.2)
                    loss_log = criterion(pred_log, y_log)
                    val_loss_log += loss_log.item()
                    
                    # Reconversion
                    y_true_raw = torch.expm1(y_log)
                    y_pred_raw = torch.expm1(pred_log)
                    
                    # 2. MAPE
                    val_mape += calculate_mape(y_true_raw, y_pred_raw).item()
                    
                    # 3. MAE Brute (Nouveau)
                    val_mae_brute += torch.mean(torch.abs(y_true_raw - y_pred_raw)).item()

            # Moyennes
            train_loss_log /= len(train_loader)
            val_loss_log /= len(val_loader)  
            val_mape /= len(val_loader)
            val_mae_brute /= len(val_loader)

            history['train_loss'].append(train_loss_log)
            history['val_loss'].append(val_loss_log)
            history['val_mape'].append(val_mape)
            
            epoch_duration = time.time() - epoch_start_time
            
            if val_mape < best_val_mape:
                best_val_mape = val_mape
                torch.save(model.state_dict(), f"best_model_patient_{val_patient}.pth")
                is_best = " 💾"
            else:
                is_best = ""

            # NOUVEL AFFICHAGE
            print(f"    Epoch {epoch+1:02d}/{epochs} [{epoch_duration:.1f}s] | Train MAE(Log): {train_loss_log:.4f} | Val MAE(Log): {val_loss_log:.4f} | Val MAE(Brute): {val_mae_brute:,.0f} ATP | Val MAPE: {val_mape:.2f}% {is_best}")
            
            scheduler.step(val_loss_log)

        print(f"  => Fin du Fold {fold_idx + 1}. Meilleure Val MAPE : {best_val_mape:.2f}%")
        fold_mapes.append(best_val_mape)

        plot_learning_curves(history, val_patient)

    # --- BILAN FINAL ---
    print("\n" + "="*80)
    print(f"[{time.strftime('%H:%M:%S')}] BILAN DE LA CV LOSO")
    print("="*80)
    avg_mape = sum(fold_mapes) / len(fold_mapes)
    print(f"SCORE FINAL -> MAPE MOYENNE : {avg_mape:.2f}%")

train_videos = get_well_directories(train_dir, pattern=WELL_DIR_PATTERN_TRAIN)
grouped_train_wells = group_by_well(train_videos)

train_and_validate_loso(grouped_train_wells, epochs=8)