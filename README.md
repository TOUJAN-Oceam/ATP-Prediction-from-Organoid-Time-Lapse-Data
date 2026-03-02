[généré par gémini]

🔬 Prédiction d'ATP par Deep Learning sur Organoïdes Dérivés de Patients
Ce projet s'inscrit dans le cadre du défi "AI in Oncology". L'objectif est de développer un modèle de Deep Learning capable de prédire la quantité d'ATP (une mesure de la production d'énergie cellulaire) à partir de vidéos time-lapse d'organoïdes sur 96 heures. Cette prédiction permet d'évaluer la réponse aux médicaments et d'accompagner le développement de thérapies personnalisées en oncologie.

📊 Le Défi des Données (Multiple Instance Learning)
Le défi technique majeur de ce projet réside dans l'hétérogénéité des échelles :

Entrée (Input) : Le jeu de données contient 92 632 vidéos de cavités (composées de 8 frames).

Sortie (Cible) : La valeur d'ATP cible est mesurée expérimentalement et n'est disponible qu'au niveau du puits (well), un puits regroupant un traitement spécifique pour un patient.

Pour résoudre ce problème de prédiction globale à partir de données fragmentées, ce projet implémente une architecture de Multiple Instance Learning (MIL) Temporelle. L'évaluation finale du modèle se fait sur la métrique MAPE (Mean Absolute Percentage Error).

🧠 Architecture du Modèle (EfficientTransformerMIL)
Le modèle traite les données sous forme de "sacs" (les puits) contenant des "instances" (les vidéos de cavités), en extrayant l'information spatiale puis temporelle :

Extraction Spatiale (L'Œil) : Un réseau EfficientNet-B0 pré-entraîné analyse chaque frame individuellement. Les couches profondes sont dégelées pour adapter la compréhension géométrique du réseau aux contours des organoïdes.

Extraction Temporelle (La Mémoire) : Un TransformerEncoder (mécanisme de Self-Attention) prend en entrée la séquence de caractéristiques de 4 frames lues au hasard pour comprendre la dynamique globale de survie ou de mort de l'organoïde, s'affranchissant ainsi du bruit chronologique.

Double Représentation Biologique (Mean + Max) : L'état temporel de la cavité est évalué via deux prisme simultanés concaténés (dimension 2560) :

La valeur moyenne (Mean) : Pour capter la tendance générale.

La valeur maximale (Max) : Pour repérer l'étincelle de vie (l'organoïde le plus brillant/actif), souvent fortement corrélée à un pic d'ATP.

Agrégation (Gated Attention) : Un mécanisme d'attention avancé attribue un poids à chaque cavité du puits selon son importance prédictive.

Régression : Un réseau dense final prédit le logarithme de l'ATP global du puits, optimisé via une SmoothL1Loss et un régularisateur L2 fort (weight_decay=1e-2).

⚙️ Stratégie d'Optimisation & Ensembling
Pour obtenir un score très compétitif sur la métrique MAPE et éviter tout risque d'overfitting, le script déploie une stratégie professionnelle :

Group K-Fold Cross-Validation : Les patients sont répartis dans 5 groupes distincts. La séparation se fait strictement par patient (0 Data Leakage) garantissant que le modèle est toujours validé sur des cellules qu'il n'a jamais vues.

Le "Conseil des IA" (Ensembling) : Lors de l'inférence, les 5 modèles générés par la CV sont chargés en mémoire. Chacun donne sa prédiction pour un puits test inconnu, et la décision finale est la médiane de leurs avis, ce qui réduit drastiquement la variance.

Accélération Matérielle : L'entraînement est optimisé pour les GPU avec le paramètre num_workers=4 et pin_memory=True dans les DataLoaders, divisant par deux le temps de calcul.

Résilience & Chunking : Intégration d'un système qui repère et compte les fichiers TIFF corrompus. Le réseau ingère les images par paquets (chunks de 32) pour éviter les saturations CUDA Out of Memory.

🚀 Installation & Prérequis (via uv)
Ce projet utilise uv pour une gestion ultra-rapide des paquets et de l'environnement virtuel. Assurez-vous d'avoir Python 3.10+ d'installé.

Créer et activer l'environnement virtuel :

Bash
uv venv
.venv\Scripts\activate   # Sur Windows
# source .venv/bin/activate  # Sur macOS/Linux
Installer PyTorch avec le support CUDA (Indispensable pour le GPU) :
Remarque : Modifiez cu121 selon la version de CUDA supportée par vos drivers NVIDIA.

Bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Installer le reste des dépendances :

Bash
uv pip install pillow matplotlib tqdm pandas numpy
📂 Structure des Dossiers
Modifiez les chemins dans la section 0. CONFIGURATION du script pour pointer vers vos dossiers. Le code s'attend à l'arborescence suivante :

Plaintext
DATASET/
├── train/              # Patients d'entraînement
│   ├── CGR0010/
│   │   ├── [well_id]_[atp]_[conc]_[date]/
│   │   │   ├── ..._1.tif
│   │   │   └── ...
├── test/               # Patients à prédire
│   ├── CGRXXXX/
│   │   ├── [well_id]_[conc]_[date]/
🕹️ Utilisation
Le comportement du script est contrôlé par des booléens situés en haut du fichier project_ATP.py :

Python
# --- BOUTONS DE CONTRÔLE ---
RUN_5_FOLD_ENSEMBLE = True  # Découpe le dataset et entraîne 5 modèles distincts (K-Fold)
NUM_FOLDS = 5               # Nombre de modèles à générer (par défaut: 5)
TRAIN_FULL_MODEL = False    # Entraîne 1 seul modèle sur 100% des données (idéal pour un test rapide)
RUN_PREDICTION = True       # Inférence sur le Test Set. Utilise automatiquement l'Ensemble s'il est activé.
Pour lancer le pipeline complet :

Bash
uv run project_ATP.py
# ou simplement : python project_ATP.py
📈 Résultats et Fichiers Générés
Selon les options activées, le script génère automatiquement dans son répertoire :

Les Modèles : Les poids des modèles entraînés (ensemble_model_fold_X.pth ou final_model_production.pth).

Le Fichier de Soumission : Un fichier .csv (submission_ATP_ensembling_median.csv ou submission_ATP.csv) parfaitement trié et formaté pour l'évaluation, contenant les valeurs brutes d'ATP prédites (pid, well_id, atp_value_raw).

Les Courbes d'Apprentissage : (Si modèle complet) Un graphique .png affichant la perte logarithmique pour suivre la santé de l'entraînement.

Bilan Sanitaire : Dans la console en fin d'exécution, un rapport détaillé des statistiques de lecture et du taux de corruption des fichiers TIFF.