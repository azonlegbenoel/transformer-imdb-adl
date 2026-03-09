# Transformer Encoder — Analyse de sentiment (IMDB)

Implémentation from scratch d'un **Transformer Encoder** pour la classification binaire de sentiment sur le dataset IMDB.  
Basé sur : *Vaswani et al., "Attention Is All You Need" (2017)*

---

## Architecture

```
Tokens  →  Embedding (d=128)  →  Encodage positionnel sinusoïdal
        →  3 × CoucheTransformer
                ├─ Pre-LayerNorm
                ├─ Multi-Head Self-Attention (4 têtes)
                └─ Feed-Forward (d_ff=512, GELU)
        →  Moyenne pondérée (hors padding)
        →  Linear(128 → 1)  →  BCEWithLogitsLoss
```

**Paramètres entraînables : ~2.5 M**

---

## Lancer l'entraînement

```bash
pip install -r requirements.txt

# Entraînement standard (avec dropout)
python train.py --epoques 10 --batch 64

# Ablation study complète (avec vs sans dropout)
python train.py --epoques 10 --batch 64 --ablation
```

## Évaluation

```bash
python evaluer.py --checkpoint meilleur_avec_dropout.pth
```

---

## Choix d'implémentation

**Initialisation Xavier** : les couches d'attention utilisent softmax (activation symétrique),
ce qui justifie Xavier plutôt que He. Xavier conserve la variance du signal dans les deux sens
de propagation avec `std = sqrt(2 / (fan_in + fan_out))`.

**Ablation study** : comparaison avec vs sans dropout (taux 0.1).  
Le dropout force le réseau à ne pas sur-spécialiser ses têtes d'attention sur des tokens précis,
ce qui améliore la généralisation.

**Connexions résiduelles** : chaque `CoucheTransformer` ajoute l'entrée à la sortie de
l'attention et du FFN. Le gradient reçoit toujours la contribution directe du chemin résiduel,
ce qui empêche sa disparition en profondeur.

---

## Résultats attendus

| Variante | Précision test |
|---|---|
| Avec dropout | ~86–88 % |
| Sans dropout | ~82–84 % |
