[généré par gémini]

# 🔬 Prédiction d'ATP par Deep Learning sur Organoïdes Dérivés de Patients

Ce projet s'inscrit dans le cadre du défi "AI in Oncology". L'objectif est de développer un modèle de Deep Learning capable de prédire la quantité d'ATP (une mesure de la production d'énergie cellulaire) à partir de vidéos time-lapse d'organoïdes sur 96 heures. Cette prédiction permet d'évaluer la réponse aux médicaments et d'accompagner le développement de thérapies personnalisées en oncologie.

## 📊 Le Défi des Données (Multiple Instance Learning)

Le défi technique majeur de ce projet réside dans l'hétérogénéité des échelles :
* **Entrée (Input) :** Le jeu de données contient 92 632 vidéos de **cavités** (composées de 8 frames).
* **Sortie (Cible) :** La valeur d'ATP cible est mesurée expérimentalement et n'est disponible qu'au niveau du **puits** (well), un puits regroupant un traitement spécifique pour un patient.

Pour résoudre ce problème de prédiction globale à partir de données fragmentées, ce projet implémente une architecture de **Multiple Instance Learning (MIL) Temporelle**. L'évaluation finale du modèle se fait sur la métrique **MAPE** (Mean Absolute Percentage Error).

## 🧠 Architecture du Modèle (`EfficientTransformerMIL`)

Le modèle traite les données sous forme de "sacs" (les puits) contenant des "instances" (les vidéos de cavités), en extrayant l'information spatiale puis temporelle :

1. **Extraction Spatiale (L'Œil) :** Un réseau `EfficientNet-B0` pré-entraîné analyse chaque frame individuellement. Les premières couches sont gelées pour éviter le sur-apprentissage sur la cohorte d'entraînement (36 patients).
2. **Extraction Temporelle (La Mémoire) :** Un `TransformerEncoder` prend en entrée la séquence de caractéristiques des 4 frames sélectionnées (0, 2, 5, 7) pour comprendre la dynamique de survie ou de mort de l'organoïde.
3. **Agrégation (Gated Attention) :** Un mécanisme d'attention avancé (Gated Attention) attribue un poids à chaque cavité du puits selon son importance prédictive.
4. **Régression :** Un réseau dense final prédit le logarithme de l'ATP global du puits, optimisé via une `SmoothL1Loss`.

## ⚙️ Fonctionnalités du Script

* **Gestion de la Mémoire GPU (Chunking) :** Les images passent dans le réseau par paquets (chunks de 32) pour éviter les erreurs `CUDA Out of Memory`.
* **Résilience aux Données :** Intégration d'un système robuste qui repère, ignore et compte les fichiers TIFF corrompus sans faire planter l'entraînement.
* **Data Augmentation :** Application de modifications aléatoires (Autocontrast, ColorJitter, Rotations, Flips) pour assurer la robustesse du modèle.
* **Early Stopping & Scheduler :** Réduction dynamique du Learning Rate et arrêt prématuré pour optimiser la convergence.

## 🚀 Installation & Prérequis (via `uv`)

Ce projet utilise **`uv`** pour une gestion ultra-rapide des paquets et de l'environnement virtuel. Assurez-vous d'avoir Python 3.10+ d'installé.

1. **Créer et activer l'environnement virtuel :**
```bash
uv venv
.venv\Scripts\activate   # Sur Windows
# source .venv/bin/activate  # Sur macOS/Linux
```
Installer PyTorch avec le support CUDA (Indispensable pour le GPU) :
Remarque : Modifiez cu121 selon la version de CUDA supportée par vos drivers NVIDIA (ex: cu118, cu124).

```Bash
uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
Installer le reste des dépendances :

```Bash
uv pip install pillow matplotlib tqdm
```
📂 Structure des Dossiers
Modifiez les chemins dans la section 0. CONFIGURATION du script pour pointer vers vos dossiers. Le code s'attend à l'arborescence suivante :

Plaintext
DATASET/
├── train/              # 36 patients
│   ├── CGR0010/
│   │   ├── [well_id]_[atp]_[conc]_[date]/
│   │   │   ├── ..._1.tif
│   │   │   └── ...
├── test/               # 15 patients
│   ├── CGRXXXX/
│   │   ├── [well_id]_[conc]_[date]/


🕹️ Utilisation
Le comportement du script est contrôlé par trois booléens situés en haut du fichier project_ATP.py. Vous pouvez les activer/désactiver selon vos besoins :


Python
RUN_CV = False             # Lance une validation croisée Leave-One-Subject-Out (LOSO)
TRAIN_FULL_MODEL = True    # Entraîne le modèle de production final sur 100% du jeu d'entraînement
RUN_PREDICTION = True      # Effectue l'inférence sur le dossier Test et génère le fichier de soumission
Pour lancer le pipeline dans votre environnement uv :

```Bash
uv run project_ATP.py
# ou simplement : python project_ATP.py
```
📈 Résultats et Fichiers Générés
Selon les options activées, le script génère automatiquement :

Les Courbes d'Apprentissage : Des fichiers learning_curve_CGRXXXX.png affichant la MAE (Log) et la MAPE de validation pour suivre la santé de l'entraînement.

Les Modèles : Les poids du meilleur modèle (final_model_production.pth).

Le Fichier de Soumission : Un fichier .csv (par défaut submission_ATP.csv) formaté pour l'évaluation, contenant les valeurs brutes d'ATP prédites (pid, well_id, atp_value_raw).

Rapport de Corruption : Dans la console, un bilan de lecture détaillant le pourcentage de fichiers TIFF corrompus rencontrés.
