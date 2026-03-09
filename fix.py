import re

with open("train.py", "r") as f:
    contenu = f.read()

contenu = contenu.replace(", verbose=True", "")

with open("train.py", "w") as f:
    f.write(contenu)

print("Corrigé !")
