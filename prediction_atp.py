import os
import glob
import re
import math
import random
import csv
import time
from collections import defaultdict
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.v2 as transforms 
import torchvision.models as models
import matplotlib.pyplot as plt

# ==========================================
# 0. CONFIGURATION & BOUTONS DE CONTRÔLE
# ==========================================
train_dir = r"C:\Users\black\Desktop\DATASET\train"  
test_dir = r"C:\Users\black\Desktop\DATASET\test"    

WELL_DIR_PATTERN_TRAIN = re.compile(
    r"^(?P<well_id>[0-9a-fA-F]+)_(?P<atp>\d+)_(?P<conc>[A-Za-z]+)_(?P<date>\d+)$"
)

WELL_DIR_PATTERN_TEST = re.compile(
    r"^(?P<well_id>[0-9a-fA-F]+)_(?P<conc>[A-Za-z]+)_(?P<date>\d+)$"
)

CORRUPTION_STATS = {
    "total_attempts": 0,
    "corrupted_files": 0
}


# --- BOUTONS DE CONTRÔLE ---
RUN_5_FOLD_ENSEMBLE = False  # OUI : C'est lui qui va générer et sauvegarder les 5 modèles !
NUM_FOLDS = 5               # Nombre de modèles / sous-groupes
TRAIN_FULL_MODEL = False    # NON : On ne veut plus de modèle unique.
RUN_PREDICTION = True       # OUI : Va charger les 5 modèles à la fin et les faire voter.

# # ==========================================
# # 1. VÉRIFICATION DU GPU
# # ==========================================
# print("="*50)
# print(f"Version de PyTorch : {torch.__version__}")
# if torch.cuda.is_available():
#     print(f"✅ GPU détecté : {torch.cuda.get_device_name(0)}")
#     print(f"✅ Nombre de GPUs : {torch.cuda.device_count()}")
#     torch.cuda.empty_cache() 
# else:
#     print("❌ ATTENTION : Aucun GPU CUDA détecté. L'apprentissage se fera sur le CPU.")
# print("="*50)


# ==========================================
# 2. PRÉPARATION DES DONNÉES & DATASET
# ==========================================
def get_well_directories(data_dir: str, pattern: re.Pattern):
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
    wells_data = defaultdict(lambda: {"cavities": [], "atp": None, "patient_id": None})
    for item in cavity_list:
        wid = item["well_id"]
        wells_data[wid]["cavities"].append(item["tif_path"])
        if "atp" in item:
            wells_data[wid]["atp"] = item["atp"]
        wells_data[wid]["patient_id"] = item["patient_id"]
    return list(wells_data.items())

class WellDatasetMIL(Dataset):
    def __init__(self, wells_list, img_size=224, is_train=True, sample_frac=1.0):
        self.wells_list = wells_list
        self.img_size = img_size
        self.is_train = is_train
        self.sample_frac = sample_frac
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

        if self.is_train:
            # ==========================================================
            # NOUVEAU : Ajout de l'amélioration du contraste et luminosité
            # ==========================================================
            self.augment = transforms.Compose([
                transforms.RandomAutocontrast(p=0.5), # Maximise le contraste 1 fois sur 2
                transforms.ColorJitter(brightness=0.2, contrast=0.2), # Légères variations
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
            
        # Remplacer 'frame_indices = [0, 2, 5, 7]' par :
        if self.is_train:
            # Choisit 4 frames au hasard (en supposant qu'il y a ~8 frames max)
            # On les trie pour garder l'ordre chronologique
            frame_indices = sorted(random.sample(range(8), 4)) 
        else:
            # En test, on garde toujours les mêmes pour être déterministe
            frame_indices = [0, 2, 5, 7]
        all_cavities_tensors = []
        
        for path in sampled_paths:
            global CORRUPTION_STATS
            CORRUPTION_STATS["total_attempts"] += 1
            
            try:
                img = Image.open(path)
                img.load() 
                
                frames = []
                for i in frame_indices:
                    img.seek(i if i < img.n_frames else img.n_frames - 1)
                    frame = transforms.functional.to_image(img).float() / 255.0
                    frame = frame.repeat(3, 1, 1) 
                    frames.append(frame)
                
                cavity_tensor = torch.stack(frames, dim=0) 
                
                if self.augment is not None:
                    cavity_tensor = self.augment(cavity_tensor)
                    
                cavity_tensor = self.normalize(cavity_tensor)
                cavity_tensor = transforms.functional.resize(cavity_tensor, (self.img_size, self.img_size))
                
                all_cavities_tensors.append(cavity_tensor)
                
            except Exception as e:
                CORRUPTION_STATS["corrupted_files"] += 1
                continue
            
        if len(all_cavities_tensors) == 0:
            dummy_tensor = torch.zeros((1, 4, 3, self.img_size, self.img_size))
            all_cavities_tensors.append(dummy_tensor)

        well_tensor = torch.stack(all_cavities_tensors, dim=0)

        if well_data["atp"] is not None:
            y = torch.tensor([math.log1p(well_data["atp"])], dtype=torch.float32)
            return well_tensor, y, well_id, well_data["patient_id"]
        else:
            return well_tensor, well_id, well_data["patient_id"]


# ==========================================
# 3. NOUVELLE ARCHITECTURE : EfficientNet + Transformer
# ==========================================
class EfficientTransformerMIL(nn.Module):
    def __init__(self):
        super(EfficientTransformerMIL, self).__init__()
        
        # 1. CNN (EfficientNet-B0)
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn = nn.Sequential(
            efficientnet.features,
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Gel des premières couches. On ne laisse que les blocs 7 et 8 apprendre.
        for name, param in self.cnn.named_parameters():
            if not ("0.7." in name or "0.8." in name):
                param.requires_grad = False
                
        # La taille des vecteurs extraits par EfficientNet-B0 est 1280
        feature_dim = 1280
        
        # --- NOUVEAU : La taille combinée (Mean + Max = 1280 + 1280 = 2560) ---
        combined_dim = feature_dim * 2 
        
        # 2. Le Cerveau Temporel (Transformer)
        # Le Transformer continue de traiter les images une par une (donc taille 1280)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 3. Gated Attention 
        # MODIFIÉ : Accepte maintenant 2560 en entrée !
        self.attention_V = nn.Sequential(nn.Linear(combined_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(combined_dim, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, 1)
        
        # 4. Régresseur
        # MODIFIÉ : Accepte maintenant 2560 en entrée !
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.squeeze(0) 
        N, S, C, H, W = x.size()
        
        x_flat = x.view(N * S, C, H, W)
        
        # CHUNKING
        chunk_size = 32
        features_list = []
        for i in range(0, x_flat.size(0), chunk_size):
            chunk = x_flat[i : i + chunk_size]
            chunk_feat = self.cnn(chunk) 
            features_list.append(chunk_feat)
            
        features = torch.cat(features_list, dim=0) # Shape: (N * S, 1280, 1, 1)
        features = features.view(N, S, 1280)       # On enlève les dimensions 1,1
        
        # Analyse temporelle avec le Transformer
        transformer_out = self.transformer(features) # Shape: (N_cavités, 4_frames, 1280)
        
        # Au lieu de prendre le dernier état comme le LSTM, on fait la moyenne des 4 frames analysées
        cavity_mean = torch.mean(transformer_out, dim=1)
        cavity_max = torch.max(transformer_out, dim=1)[0]

        cavity_embeddings = torch.cat([cavity_mean, cavity_max], dim=1)
        
        # Gated Attention
        A_V = self.attention_V(cavity_embeddings)
        A_U = self.attention_U(cavity_embeddings)
        attn_scores = self.attention_weights(A_V * A_U)
        attn_weights = torch.softmax(attn_scores, dim=0)
        
        well_embedding = torch.sum(cavity_embeddings * attn_weights, dim=0, keepdim=True)
        out = self.regressor(well_embedding)
        
        return out




# ==========================================
# 5. ENTRAÎNEMENT ENSEMBLE (5-FOLD)
# ==========================================
def train_kfold_ensemble(grouped_wells_list, k=5, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 Démarrage de l'entraînement ENSEMBLE ({k}-Folds)...")

    # On récupère tous les patients uniques et on les mélange
    patients = list(set([data["patient_id"] for _, data in grouped_wells_list]))
    random.seed(42) # Seed fixe pour toujours avoir le même découpage
    random.shuffle(patients)
    
    # On calcule la taille d'un bloc (ex: 731 / 5 = ~146 patients par bloc)
    fold_size = len(patients) // k
    
    for fold in range(k):
        print("\n" + "="*60)
        print(f"🔄 DÉMARRAGE DU MODÈLE {fold+1} SUR {k}")
        print("="*60)
        
        # Découpage des patients pour ce Fold
        if fold < k - 1:
            val_patients = patients[fold * fold_size : (fold + 1) * fold_size]
        else:
            val_patients = patients[fold * fold_size :] # Le dernier prend le reste
            
        train_data = [item for item in grouped_wells_list if item[1]["patient_id"] not in val_patients]
        val_data = [item for item in grouped_wells_list if item[1]["patient_id"] in val_patients]

        train_dataset = WellDatasetMIL(train_data, is_train=True, sample_frac=1.0)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        

        val_dataset = WellDatasetMIL(val_data, is_train=False) 
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        model = EfficientTransformerMIL().to(device)
        criterion = nn.SmoothL1Loss(beta=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # --- TRAIN ---
            model.train()
            train_loss_log = 0.0
            for x, y, _, _ in tqdm(train_loader, desc=f"M{fold+1} - Train Ep {epoch+1:02d}", leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                predictions = model(x)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()
                train_loss_log += loss.item()

            # --- VAL ---
            model.eval()
            val_loss_log = 0.0
            with torch.no_grad():
                for x, y_log, _, _ in val_loader: # Pas de tqdm ici pour garder la console propre
                    x, y_log = x.to(device), y_log.to(device)
                    pred_log = model(x)
                    val_loss_log += criterion(pred_log, y_log).item()
                    
            train_loss_log /= len(train_loader)
            val_loss_log /= max(1, len(val_loader))
            epoch_duration = time.time() - epoch_start_time
            
            # Sauvegarde du meilleur modèle pour CE fold
            if val_loss_log < best_val_loss:
                best_val_loss = val_loss_log
                torch.save(model.state_dict(), f"ensemble_model_fold_{fold+1}.pth")
                is_best = " 💾"
            else:
                is_best = ""

            print(f"    Epoch {epoch+1:02d}/{epochs} [{epoch_duration:.1f}s] | Train MAE: {train_loss_log:.4f} | Val MAE: {val_loss_log:.4f}{is_best}")
            scheduler.step()




# ==========================================
# 6. PRÉDICTION ENSEMBLE SUR TEST SET
# ==========================================
def predict_ensemble_test_set(test_dir, pattern, output_csv="submission_ATP.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔮 Démarrage de l'inférence avec LE CONSEIL DES IA (Ensemble)...")

    test_videos = get_well_directories(test_dir, pattern=pattern)
    if not test_videos:
        print("❌ Aucune vidéo trouvée dans le dossier test !")
        return
        
    grouped_test_wells = group_by_well(test_videos)
    print(f"Nombre de puits à prédire : {len(grouped_test_wells)}")

    test_dataset = WellDatasetMIL(grouped_test_wells, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # RECHERCHE DES 5 MODÈLES
    model_paths = glob.glob("ensemble_model_fold_*.pth")
    if not model_paths:
        print("❌ Aucun modèle 'ensemble_model_fold_X.pth' trouvé ! Avez-vous lancé l'entraînement ?")
        return
        
    print(f"🧠 {len(model_paths)} cerveaux (modèles) chargés pour le vote !")
    
    # Chargement des modèles en mémoire (carte graphique)
    models_list = []
    for path in model_paths:
        m = EfficientTransformerMIL().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        models_list.append(m)

    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Vote en cours..."):
            x = batch[0].to(device)
            well_id = batch[1][0]
            pid = batch[2][0]

            # Chaque modèle donne sa prédiction
            raw_predictions = []
            for m in models_list:
                pred_log = m(x)
                pred_raw = torch.expm1(pred_log).item()
                raw_predictions.append(pred_raw)

            # LA MAGIE DE L'ENSEMBLING : On fait la moyenne de leurs avis !
            final_pred = sum(raw_predictions) / len(raw_predictions)
            sorted(raw_predictions)
            print(int(len(raw_predictions)/2))
            final_pred = raw_predictions[int(len(raw_predictions)/2)]

            results.append({
                "pid": pid, 
                "well_id": well_id, 
                "atp_value_raw": final_pred
            })

    # TRÈS IMPORTANT : Le tri pour votre script Jupyter !
    results = sorted(results, key=lambda d: (d['pid'], d['well_id']))

    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['pid', 'well_id', 'atp_value_raw']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Fichier d'Ensemble généré et TRIÉ avec succès : {output_csv}")




# ==========================================
# 4. FONCTIONS DE GRAPHIQUES & VALIDATION
# ==========================================
def calculate_mape(y_true_raw, y_pred_raw):
    return torch.mean(torch.abs((y_true_raw - y_pred_raw) / y_true_raw)) * 100

def plot_learning_curves(history, patient_id):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('MAE (Log)', color=color, fontsize=12)
    ax1.plot(epochs, history['train_loss'], color='darkblue', label='Train MAE', marker='o')
    ax1.plot(epochs, history['val_loss'], color='deepskyblue', label='Val MAE', marker='s')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Val MAPE (%)', color=color, fontsize=12)
    ax2.plot(epochs, history['val_mape'], color='crimson', label='Val MAPE', marker='^', linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    plt.title(f'Courbes d\'apprentissage - Patient : {patient_id}', fontsize=14)
    fig.tight_layout()  
    plt.savefig(f'learning_curve_{patient_id}.png', dpi=300)
    plt.close()

def plot_full_training_curve(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], color='darkblue', label='Train MAE (Log)', marker='o', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MAE (Log)', color='tab:blue', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.title('Courbe d\'apprentissage - Entraînement Final (100% des données)', fontsize=14)
    plt.tight_layout()  
    plt.savefig('learning_curve_FINAL_PRODUCTION.png', dpi=300)
    plt.close()
    print("    📊 Graphique de production sauvegardé sous : learning_curve_FINAL_PRODUCTION.png")

def train_and_validate_loso(grouped_wells_list, epochs=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] Démarrage CV LOSO sur : {device}")

    patients = list(set([data["patient_id"] for _, data in grouped_wells_list]))
    fold_mapes = []

    for fold_idx, val_patient in enumerate(patients):
        print(f"\n[{time.strftime('%H:%M:%S')}] === FOLD {fold_idx + 1}/{len(patients)} | Patient en Validation : {val_patient} ===")
        
        train_data = [item for item in grouped_wells_list if item[1]["patient_id"] != val_patient]
        val_data = [item for item in grouped_wells_list if item[1]["patient_id"] == val_patient]

        train_dataset = WellDatasetMIL(train_data, is_train=True, sample_frac=1.0)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        val_dataset = WellDatasetMIL(val_data, is_train=False) 
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        model = EfficientTransformerMIL().to(device)
        criterion = nn.SmoothL1Loss(beta=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

        best_val_mape = float('inf')
        epochs_without_improvement = 0
        patience_early_stopping = 3
        history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            model.train()
            train_loss_log = 0.0
            for x, y, _, _ in tqdm(train_loader, desc=f"Train Epoch {epoch+1:02d}", leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                predictions = model(x)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()
                train_loss_log += loss.item()

            model.eval()
            val_loss_log = 0.0
            val_mape = 0.0
            val_mae_brute = 0.0
            
            with torch.no_grad():
                for x, y_log, _, _ in tqdm(val_loader, desc=f"Val Epoch {epoch+1:02d}", leave=False):
                    x, y_log = x.to(device), y_log.to(device)
                    pred_log = model(x)
                    
                    val_loss_log += criterion(pred_log, y_log).item()
                    y_true_raw = torch.expm1(y_log)
                    y_pred_raw = torch.expm1(pred_log)
                    
                    val_mape += calculate_mape(y_true_raw, y_pred_raw).item()
                    val_mae_brute += torch.mean(torch.abs(y_true_raw - y_pred_raw)).item()

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
                epochs_without_improvement = 0
            else:
                is_best = ""
                epochs_without_improvement += 1

            print(f"    Epoch {epoch+1:02d}/{epochs} [{epoch_duration:.1f}s] | Train MAE: {train_loss_log:.4f} | Val MAE: {val_loss_log:.4f} | Val MAPE: {val_mape:.2f}% {is_best}")
            scheduler.step(val_loss_log)
            
            if epochs_without_improvement >= patience_early_stopping:
                print(f"    🛑 Early Stopping !")
                break

        print(f"  => Fin du Fold {fold_idx + 1}. Meilleure Val MAPE : {best_val_mape:.2f}%")
        fold_mapes.append(best_val_mape)
        plot_learning_curves(history, val_patient)

    print("\n" + "="*80)
    print(f"[{time.strftime('%H:%M:%S')}] BILAN DE LA CV LOSO : {sum(fold_mapes) / len(fold_mapes):.2f}%")


# ==========================================
# 5. ENTRAÎNEMENT FINAL (SUR 100% DES DONNÉES)
# ==========================================
def train_full_model(grouped_wells_list, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 Démarrage de l'entraînement FINAL sur TOUTES les données...")

    train_dataset = WellDatasetMIL(grouped_wells_list, is_train=True, sample_frac=1.0)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = EfficientTransformerMIL().to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_train_loss = float('inf')
    history = {'train_loss': []}

    model.train()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss_log = 0.0
        
        for x, y, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            train_loss_log += loss.item()

        train_loss_log /= len(train_loader)
        history['train_loss'].append(train_loss_log)
        
        epoch_duration = time.time() - epoch_start_time
        
        if train_loss_log < best_train_loss:
            best_train_loss = train_loss_log
            torch.save(model.state_dict(), "final_model_production.pth")
            is_best = " 💾 (Meilleur modèle sauvegardé !)"
        else:
            is_best = ""

        print(f"    Epoch {epoch+1:02d}/{epochs} [{epoch_duration:.1f}s] | Train MAE(Log): {train_loss_log:.4f}{is_best}")
        scheduler.step() 

    print("\n  => Entraînement terminé !")
    plot_full_training_curve(history)


# ==========================================
# 6. PRÉDICTION SUR LE TEST SET & EXPORT CSV
# ==========================================
def predict_test_set(model_path, test_dir, pattern, output_csv="submission_ATP.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔮 Démarrage de l'inférence sur le dossier Test...")

    test_videos = get_well_directories(test_dir, pattern=pattern)
    if not test_videos:
        print("❌ Aucune vidéo trouvée dans le dossier test ! Vérifiez le chemin ou la Regex.")
        return
        
    grouped_test_wells = group_by_well(test_videos)
    print(f"Nombre de puits à prédire : {len(grouped_test_wells)}")

    test_dataset = WellDatasetMIL(grouped_test_wells, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = EfficientTransformerMIL().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Prédiction en cours"):
            x = batch[0].to(device)
            well_id = batch[1][0]
            pid = batch[2][0]

            pred_log = model(x)
            pred_raw = torch.expm1(pred_log).item()

            results.append({
                "pid": pid, 
                "well_id": well_id, 
                "atp_value_raw": pred_raw
            })

    results = sorted(results, key=lambda d: (d['pid'], d['well_id']))

    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['pid', 'well_id', 'atp_value_raw']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Fichier généré avec succès : {output_csv}")


# ==========================================
# 7. EXÉCUTION DU WORKFLOW
# ==========================================
if __name__ == "__main__":
    
    # --- 1. ENTRAÎNEMENT ENSEMBLE (5 FOLDS) ---
    if RUN_5_FOLD_ENSEMBLE:
        train_videos = get_well_directories(train_dir, pattern=WELL_DIR_PATTERN_TRAIN)
        grouped_train_wells = group_by_well(train_videos)
        train_kfold_ensemble(grouped_train_wells, k=NUM_FOLDS, epochs=10)
        
    # --- 2. ENTRAÎNEMENT DU MODÈLE UNIQUE (FULL MODEL) ---
    if TRAIN_FULL_MODEL:
        train_videos = get_well_directories(train_dir, pattern=WELL_DIR_PATTERN_TRAIN)
        grouped_train_wells = group_by_well(train_videos)
        train_full_model(grouped_train_wells, epochs=12)


    RUN_5_FOLD_ENSEMBLE = True
    # --- 3. PRÉDICTION ---
    if RUN_PREDICTION:
        # Si on est en mode Ensembling, on utilise le Conseil des IA
        if RUN_5_FOLD_ENSEMBLE:
            predict_ensemble_test_set(
                test_dir=test_dir, 
                pattern=WELL_DIR_PATTERN_TEST,
                output_csv="submission_ATP_ensembling_median.csv"
            )
        # Sinon, on utilise le modèle unique classique
        else:
            predict_test_set(
                model_path="final_model_production.pth", 
                test_dir=test_dir, 
                pattern=WELL_DIR_PATTERN_TEST,
                output_csv="submission_ATP.csv"
            )
        
    # --- BILAN CORRUPTION ---
    print("\n" + "="*80)
    print("📊 BILAN DE LECTURE DES FICHIERS (CORRUPTION)")
    print("="*80)
    total = CORRUPTION_STATS["total_attempts"]
    corrompus = CORRUPTION_STATS["corrupted_files"]
    if total > 0:
        pourcentage = (corrompus / total) * 100
        print(f"Fichiers tentés    : {total:,}".replace(',', ' '))
        print(f"Fichiers corrompus : {corrompus:,}".replace(',', ' '))
        print(f"Taux de corruption : {pourcentage:.3f} %")
    else:
        print("Aucun fichier n'a été lu.")
    print("="*80)