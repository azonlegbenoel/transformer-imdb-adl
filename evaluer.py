"""
Évaluation du modèle sur IMDB (test set).
Usage :
    python evaluer.py --checkpoint meilleur_avec_dropout.pth
"""

import torch
import json, argparse
from torch.utils.data import DataLoader
from train import telecharger_imdb, charger_imdb, construire_vocab, DatasetIMDB
from model import TransformerClassifieur


def evaluer(chemin_checkpoint, max_len=256, batch=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    telecharger_imdb()
    donnees = charger_imdb()

    textes_train = [t for t, _ in donnees['train']]
    textes_test  = [t for t, _ in donnees['test']]
    labels_test  = [l for _, l in donnees['test']]

    vocab        = construire_vocab(textes_train)
    taille_vocab = len(vocab)

    ds_test     = DatasetIMDB(textes_test, labels_test, vocab, max_len)
    test_loader = DataLoader(ds_test, batch_size=batch, shuffle=False, num_workers=2)

    modele = TransformerClassifieur(taille_vocab=taille_vocab).to(device)
    modele.load_state_dict(torch.load(chemin_checkpoint, map_location=device))
    modele.eval()

    correct, total = 0, 0
    vp, vn, fp, fn = 0, 0, 0, 0   # vrais/faux positifs/négatifs

    with torch.no_grad():
        for ids, masque, labels in test_loader:
            ids, masque, labels = ids.to(device), masque.to(device), labels.to(device)
            preds = (modele(ids, masque) > 0).long()
            correct += preds.eq(labels.long()).sum().item()
            total   += labels.size(0)
            vp += ((preds == 1) & (labels == 1)).sum().item()
            vn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    acc       = 100.0 * correct / total
    precision = 100.0 * vp / (vp + fp) if (vp + fp) > 0 else 0
    rappel    = 100.0 * vp / (vp + fn) if (vp + fn) > 0 else 0
    f1        = 2 * precision * rappel / (precision + rappel) if (precision + rappel) > 0 else 0

    print(f"\nCheckpoint  : {chemin_checkpoint}")
    print(f"Précision globale : {acc:.2f}%")
    print(f"Précision (pos)   : {precision:.2f}%")
    print(f"Rappel    (pos)   : {rappel:.2f}%")
    print(f"F1-score          : {f1:.2f}%")

    resultats = {"accuracy": round(acc, 2), "precision": round(precision, 2),
                 "rappel": round(rappel, 2), "f1": round(f1, 2)}
    with open("resultats_eval.json", "w") as f:
        json.dump(resultats, f, indent=2)
    print("Résultats sauvegardés dans resultats_eval.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="meilleur_avec_dropout.pth")
    parser.add_argument("--max_len",    type=int, default=256)
    args = parser.parse_args()
    evaluer(args.checkpoint, args.max_len)
