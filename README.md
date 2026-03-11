# Transformer Encoder — Analyse de Sentiment IMDB

**TP n°2 — Advanced Deep Learning · ENEAM / ISE 3**  
Étudiant : AZONLEGBE Noel Junior Azonsou  
Encadrant : Rodéo Oswald Y. TOHA  
Année : 2025-2026

---

## Ce que fait ce projet

Implémentation **from scratch** d'un Transformer Encoder pour la classification binaire de sentiment sur le dataset IMDB (avis positifs / négatifs). Aucun modèle pré-entraîné, aucune couche importée de HuggingFace — tout est codé à la main : attention multi-têtes, encodage positionnel, FFN, résidus, normalisation.

**Résultat principal : 87.04% de précision test** (époque 2, run de 20 époques)  
**Ablation dropout : +2.55 pts** avec dropout vs sans dropout (15 époques)

---

## Structure du dépôt

```
transformer-imdb-adl/
├── model.py                      # Architecture complète from scratch
├── train.py                      # Entraînement + ablation study
├── evaluer.py                    # Évaluation sur données externes
├── requirements.txt
├── meilleur_avec_dropout.pth     # Checkpoint meilleur modèle
├── meilleur_sans_dropout.pth     # Checkpoint ablation sans dropout
├── historique_avec_dropout.json  # Courbes d'entraînement
├── historique_sans_dropout.json
└── resultats_ablation.json       # Résultats comparatifs
```

---

## Lancer le projet (Google Colab)

```bash
# 1. Cloner
git clone https://github.com/azonlegbenoel/transformer-imdb-adl.git
cd transformer-imdb-adl

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Entraînement principal (20 époques)
python train.py --epoques 20 --batch 64

# 4. Ablation study (avec vs sans dropout, 15 époques)
python train.py --epoques 15 --batch 64 --ablation

# 5. Évaluer le modèle sauvegardé
python evaluer.py --checkpoint meilleur_avec_dropout.pth
```

---

## Architecture

Le modèle est dans `model.py`. Voici les blocs implémentés manuellement :

| Bloc | Détail |
|------|--------|
| `EncodagePositionnel` | Sinus/cosinus, formule PE(pos, 2i) = sin(pos / 10000^(2i/d)) |
| `AttentionMultiTete` | 4 têtes, d_k = 32, scaled dot-product, masque de padding |
| `CoucheTransformer` | Pre-LayerNorm, résidus, FFN 128→512→128, GELU, dropout |
| `TransformerClassifieur` | 3 couches, pooling pondéré sur tokens réels, Linear(128→1) |

**Hyperparamètres :**
- d_model = 128, 4 têtes, d_ff = 512, 3 couches
- Vocabulaire : 20 000 tokens, longueur max : 256
- Optimiseur : Adam (lr=3e-4, weight_decay=1e-4)
- Scheduler : ReduceLROnPlateau (factor=0.5, patience=3)
- Initialisation : Xavier uniform (std = sqrt(2 / (fan_in + fan_out)))
- ~3.15 M paramètres entraînables

---

## Résultats

### Run principal — 20 époques

| Époque | Acc. Train | Acc. Test |
|--------|-----------|-----------|
| 1      | 83.3%     | 80.1%     |
| **2**  | **97.1%** | **87.04%** ← meilleur |
| 5      | 99.6%     | 85.3%     |
| 10     | 99.9%     | 84.7%     |
| 20     | 99.9%     | 82.6%     |

Le modèle atteint son pic à l'époque 2, puis surapprend progressivement.

### Ablation Study — 15 époques (dropout 0.1 vs 0.0)

| Époque | Test avec dropout | Test sans dropout |
|--------|-----------------|-----------------|
| 1      | 79.8%           | 80.0%           |
| **2**  | **86.6%**       | **84.0%**       |
| 7      | 84.6%           | 82.0%           |
| 15     | 83.8%           | 80.4%           |

**Gain net du dropout : +2.55 pts.**  
Sans dropout, la loss d'entraînement tombe à 0.0000 dès l'époque 4 (mémorisation totale) mais la précision test chute continuellement.

---

## Réponses aux questions du TP

### Q1 — Vanishing gradient

Mon modèle gère ce problème via 3 mécanismes combinés :

1. **Connexions résiduelles** : chaque couche calcule `sortie = x + F(x)`. La dérivée vaut `1 + dF/dx` — le "1" empêche le gradient de s'écraser à zéro, même si F est mal initialisé.

2. **Pre-LayerNorm** : la normalisation est appliquée *avant* chaque sous-bloc. Les activations restent dans une plage contrôlée à chaque étape, le gradient ne diverge pas.

3. **Attention directe** : n'importe quel token peut influencer n'importe quel autre en une seule étape — pas de chaîne de couches récurrentes, donc pas d'écrasement cumulatif.

### Q2 — Activation et convergence

J'utilise **GELU** dans la FFN. Comparé à ReLU :

- ReLU coupe à zéro tous les neurones avec entrée négative → des neurones "morts" (gradient = 0 permanent, ~30-40% sur des couches larges).
- GELU laisse passer un signal faible même pour les valeurs négatives (GELU(x) ≈ x × Phi(x)). Aucun neurone ne se bloque définitivement.

Sur mes courbes, GELU donne une descente de la loss régulière dès l'époque 1, sans les plateaux brusques qu'on observe avec ReLU.

---

## Reproductibilité

- GPU : Google Colab — Tesla T4
- Seed : `torch.manual_seed(42)`
- PyTorch >= 1.12
- Le checkpoint `meilleur_avec_dropout.pth` est inclus dans le dépôt
