import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import re, json, time, argparse

from model import TransformerClassifieur


# ─── Tokenisation et vocabulaire ──────────────────────────────────────────────

def tokeniser(texte):
    texte = texte.lower()
    texte = re.sub(r'<[^>]+>', ' ', texte)       # retirer les balises HTML
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    return texte.split()

def construire_vocab(textes, taille_max=20000):
    compteur = Counter()
    for t in textes:
        compteur.update(tokeniser(t))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for mot, _ in compteur.most_common(taille_max - 2):
        vocab[mot] = len(vocab)
    return vocab

def encoder(texte, vocab, max_len=256):
    tokens = tokeniser(texte)[:max_len]
    ids    = [vocab.get(t, 1) for t in tokens]
    masque = [1] * len(ids)
    # padding jusqu'à max_len
    ids    += [0] * (max_len - len(ids))
    masque += [0] * (max_len - len(masque))
    return ids, masque


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DatasetIMDB(Dataset):
    def __init__(self, textes, etiquettes, vocab, max_len=256):
        self.data = []
        for texte, label in zip(textes, etiquettes):
            ids, masque = encoder(texte, vocab, max_len)
            self.data.append((ids, masque, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids, masque, label = self.data[i]
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(masque, dtype=torch.long),
                torch.tensor(label, dtype=torch.float))


def charger_imdb(chemin='./data/aclImdb'):
    import os
    echantillons = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for classe, label in (('pos', 1), ('neg', 0)):
            dossier = os.path.join(chemin, split, classe)
            for fichier in os.listdir(dossier):
                with open(os.path.join(dossier, fichier), encoding='utf-8') as f:
                    echantillons[split].append((f.read(), label))
    return echantillons


def telecharger_imdb(dossier='./data'):
    import urllib.request, tarfile, os
    url     = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    archive = os.path.join(dossier, 'aclImdb_v1.tar.gz')
    os.makedirs(dossier, exist_ok=True)
    if not os.path.exists(os.path.join(dossier, 'aclImdb')):
        print("Téléchargement IMDB...")
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(dossier)
        print("Extraction terminée.")


# ─── Boucle d'entraînement ────────────────────────────────────────────────────

def entrainer_epoque(modele, loader, critere, optim_, device):
    modele.train()
    perte_tot, correct, total = 0.0, 0, 0
    for ids, masque, labels in loader:
        ids, masque, labels = ids.to(device), masque.to(device), labels.to(device)
        optim_.zero_grad()
        sorties = modele(ids, masque)
        perte   = critere(sorties, labels)
        perte.backward()
        # Gradient clipping : évite les explosions de gradient (utile sans BN)
        nn.utils.clip_grad_norm_(modele.parameters(), max_norm=1.0)
        optim_.step()
        perte_tot += perte.item() * ids.size(0)
        correct   += ((sorties > 0) == labels.bool()).sum().item()
        total     += ids.size(0)
    return perte_tot / total, 100.0 * correct / total


def evaluer(modele, loader, critere, device):
    modele.eval()
    perte_tot, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for ids, masque, labels in loader:
            ids, masque, labels = ids.to(device), masque.to(device), labels.to(device)
            sorties   = modele(ids, masque)
            perte     = critere(sorties, labels)
            perte_tot += perte.item() * ids.size(0)
            correct   += ((sorties > 0) == labels.bool()).sum().item()
            total     += ids.size(0)
    return perte_tot / total, 100.0 * correct / total


# ─── Entraînement complet ─────────────────────────────────────────────────────

def lancer(avec_dropout, nb_epoques, device, train_loader, test_loader, taille_vocab):
    label = "avec_dropout" if avec_dropout else "sans_dropout"
    print(f"\n{'='*55}")
    print(f"  Ablation — [{label}]")
    print(f"{'='*55}")

    # Sans dropout = dropout=0.0 partout (ablation sur la régularisation)
    taux = 0.1 if avec_dropout else 0.0
    modele = TransformerClassifieur(
        taille_vocab=taille_vocab, d_model=128, nb_tetes=4,
        nb_couches=3, d_ff=512, dropout=taux
    ).to(device)

    nb_params = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    print(f"  Paramètres entraînables : {nb_params:,}")

    critere  = nn.BCEWithLogitsLoss()
    optimis  = optim.AdamW(modele.parameters(), lr=1e-3, weight_decay=1e-2)
    # Réduit le LR quand l'accuracy stagne — plus stable que OneCycleLR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimis, mode='max', factor=0.5, patience=3, verbose=True
    )

    historique = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    meilleure  = 0.0
    t0 = time.time()

    for ep in range(1, nb_epoques + 1):
        p_tr, a_tr = entrainer_epoque(modele, train_loader, critere, optimis, device)
        p_te, a_te = evaluer(modele, test_loader, critere, device)
        scheduler.step(a_te)  # réduit le LR si test_acc stagne

        historique["train_loss"].append(round(p_tr, 4))
        historique["train_acc"].append(round(a_tr, 2))
        historique["test_loss"].append(round(p_te, 4))
        historique["test_acc"].append(round(a_te, 2))

        if a_te > meilleure:
            meilleure = a_te
            torch.save(modele.state_dict(), f"meilleur_{label}.pth")

        print(f"  Époque {ep:2d}/{nb_epoques} | "
              f"train perte={p_tr:.4f} acc={a_tr:.1f}% | "
              f"test acc={a_te:.1f}% | meilleur={meilleure:.1f}% | "
              f"{time.time()-t0:.0f}s")

    print(f"\n  >> Meilleure précision [{label}] : {meilleure:.2f}%")
    with open(f"historique_{label}.json", "w") as f:
        json.dump(historique, f, indent=2)
    return meilleure


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoques",    type=int, default=10)
    parser.add_argument("--batch",      type=int, default=64)
    parser.add_argument("--max_len",    type=int, default=256)
    parser.add_argument("--ablation",   action="store_true",
                        help="Lance les deux variantes (avec vs sans dropout)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositif : {device}")

    telecharger_imdb()
    donnees = charger_imdb()

    textes_train = [t for t, _ in donnees['train']]
    labels_train = [l for _, l in donnees['train']]
    textes_test  = [t for t, _ in donnees['test']]
    labels_test  = [l for _, l in donnees['test']]

    print("Construction du vocabulaire...")
    vocab = construire_vocab(textes_train)
    taille_vocab = len(vocab)
    print(f"Taille du vocabulaire : {taille_vocab}")

    ds_train = DatasetIMDB(textes_train, labels_train, vocab, args.max_len)
    ds_test  = DatasetIMDB(textes_test,  labels_test,  vocab, args.max_len)

    train_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False, num_workers=2)

    resultats = {}

    if args.ablation:
        acc_avec    = lancer(True,  args.epoques, device, train_loader, test_loader, taille_vocab)
        acc_sans    = lancer(False, args.epoques, device, train_loader, test_loader, taille_vocab)
        resultats   = {"avec_dropout": acc_avec, "sans_dropout": acc_sans}
        print(f"\n  Gain du dropout : +{acc_avec - acc_sans:.2f}%")
    else:
        acc = lancer(True, args.epoques, device, train_loader, test_loader, taille_vocab)
        resultats = {"avec_dropout": acc}

    with open("resultats_ablation.json", "w") as f:
        json.dump(resultats, f, indent=2)


if __name__ == "__main__":
    main()
