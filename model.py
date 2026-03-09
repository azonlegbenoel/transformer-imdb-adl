import torch
import torch.nn as nn
import math


# Initialisation Xavier : adaptée aux activations symétriques (softmax, tanh).
# std = sqrt(2 / (fan_in + fan_out)) — stabilise la variance en forward ET backward.
# On choisit Xavier et non He car ReLU n'est pas utilisé dans les couches d'attention.
def xavier_init(module):
    if isinstance(module, nn.Linear):
        fan_in  = module.weight.size(1)
        fan_out = module.weight.size(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(module.weight, 0.0, std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, 0.0, 0.01)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class EncodagePositionnel(nn.Module):
    # Encodage sinusoïdal fixe (non appris) — Vaswani et al. 2017
    # PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class AttentionMultiTete(nn.Module):
    # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
    # La division par sqrt(d_k) évite la saturation du softmax quand d_k est grand
    # (produits scalaires qui explosent => gradients quasi nuls).
    def __init__(self, d_model, nb_tetes, dropout=0.1):
        super().__init__()
        assert d_model % nb_tetes == 0
        self.nb_tetes = nb_tetes
        self.d_k      = d_model // nb_tetes
        self.echelle  = math.sqrt(self.d_k)

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, masque=None):
        B, T, D = x.shape
        h, d_k  = self.nb_tetes, self.d_k

        # Projection + découpage en têtes
        Q = self.Wq(x).view(B, T, h, d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, h, d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, h, d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.echelle
        if masque is not None:
            scores = scores.masked_fill(masque == 0, float('-inf'))

        poids = self.dropout(torch.softmax(scores, dim=-1))
        ctx   = torch.matmul(poids, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(ctx)


class CoucheTransformer(nn.Module):
    # Pre-LayerNorm (avant attention et FFN) : meilleure stabilité des gradients
    # que le post-LN original — le signal résiduel passe sans normalisation intermédiaire.
    def __init__(self, d_model, nb_tetes, d_ff, dropout=0.1):
        super().__init__()
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.attn    = AttentionMultiTete(d_model, nb_tetes, dropout)
        # GELU à la place de ReLU : gradient plus doux, empiriquement meilleur en NLP
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, masque=None):
        # Connexion résiduelle : dL/dx = dL/d(F(x)+x) contient toujours dL/dx_court-circuit
        # => le gradient ne peut pas disparaître complètement, même avec N couches profondes.
        x = x + self.attn(self.norm1(x), masque)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerClassifieur(nn.Module):
    def __init__(self, taille_vocab, d_model=128, nb_tetes=4,
                 nb_couches=3, d_ff=512, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding  = nn.Embedding(taille_vocab, d_model, padding_idx=0)
        self.pos_enc    = EncodagePositionnel(d_model, max_len, dropout)
        self.couches    = nn.ModuleList([
            CoucheTransformer(d_model, nb_tetes, d_ff, dropout)
            for _ in range(nb_couches)
        ])
        self.norm       = nn.LayerNorm(d_model)
        self.classif    = nn.Linear(d_model, 1)
        self.dropout    = nn.Dropout(dropout)

        self.apply(xavier_init)

    def forward(self, x, masque=None):
        out = self.pos_enc(self.embedding(x))

        masque_attn = masque.unsqueeze(1).unsqueeze(2) if masque is not None else None
        for couche in self.couches:
            out = couche(out, masque_attn)

        out = self.norm(out)

        # Moyenne pondérée sur les tokens réels (exclut le padding)
        if masque is not None:
            m   = masque.unsqueeze(-1).float()
            out = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            out = out.mean(1)

        return self.classif(self.dropout(out)).squeeze(-1)
