# Détection de pièces euro — Projet Analyse d'Image

> Master 1 Vision — Université Paris Cité 2025/2026  
> Yacine Benfatma · Ghilas Tidjet

Système de vision par ordinateur capable de **détecter automatiquement des pièces euro** dans une image, d'identifier leur valeur individuelle et de calculer le montant total.

---

## Table des matières

- [Aperçu](#aperçu)
- [Pipeline](#pipeline)
- [Structure du dépôt](#structure-du-dépôt)
- [Lancer sur Google Colab](#lancer-sur-google-colab)
- [Dataset](#dataset)
- [Résultats](#résultats)
- [Paramètres](#paramètres)

---

## Aperçu

Le système prend en entrée une photo de pièces posées sur un fond et retourne :
- le **nombre de pièces** détectées
- la **valeur identifiée** pour chacune
- le **montant total** en euros

**Approche :** transformée de Hough pour la détection des cercles + analyse HSV (centre vs couronne) pour la classification couleur + comparaison des rayons normalisés aux diamètres officiels euro.

---

## Pipeline

```
Image
  │
  ├─ Redimensionnement (max 900px)
  ├─ Niveaux de gris + CLAHE
  ├─ Flou médian (7×7)
  │
  ├─ HoughCircles → liste de (cx, cy, r)
  │
  ├─ Pour chaque cercle :
  │     ├─ Analyse HSV zone centrale vs couronne
  │     ├─ → famille : bimetal / gold / copper
  │     └─ → valeur : comparaison rayon normalisé aux diamètres euro
  │
  └─ Somme des valeurs → montant total
```

---

## Structure du dépôt

```
.
├── ProjetAnalyseImage.ipynb            # Notebook Google Colab
├── rapport/
│   └── cr_projet_image.pdf
├── dataset/
│   ├── images/               # 1.jpg, 2.jpg, ..., 14.jpg
│   └── annotations.csv       # filename, count, total
└── README.md
```

---

## Lancer sur Google Colab

### 1. Ouvrir le notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MhLjqpWxfX4C1JxRX1v9EgYcEhkKDtIu?usp=sharing)

### 2. Uploader une image et lancer la détection

```python
from google.colab import files
uploaded = files.upload()  # sélectionner ton image

img_path = list(uploaded.keys())[0]
bgr = cv2.imread(img_path)

result = detect_coins(bgr, debug=True)
```

### 3. Évaluer sur tout le dataset

```python
import pandas as pd
import os

df = pd.read_csv("dataset/annotations.csv")
results = []

for _, row in df.iterrows():
    bgr = cv2.imread(f"dataset/images/{row['filename']}")
    res = detect_coins(bgr, debug=False, param2=45)
    results.append({
        "filename":   row["filename"],
        "gt_count":   row["count"],   "pred_count": res["count"],
        "gt_total":   row["total"],   "pred_total": round(res["total"], 2),
        "count_ok":   res["count"] == row["count"],
        "total_ok":   abs(res["total"] - row["total"]) < 0.01,
    })
    status = "✅" if results[-1]["count_ok"] and results[-1]["total_ok"] else "❌"
    print(f"{status} {row['filename']:10s} | count {row['count']}→{res['count']} | total {row['total']}→{res['total']:.2f}€")

df_res = pd.DataFrame(results)
print(f"\nCount accuracy : {df_res['count_ok'].mean()*100:.1f}%")
print(f"Total accuracy : {df_res['total_ok'].mean()*100:.1f}%")
print(f"MAE total      : {(df_res['pred_total']-df_res['gt_total']).abs().mean():.2f}€")
```

---

## Installation locale

> Python 3.8+ requis.

---

## Dataset

14 images acquises avec un smartphone dans des conditions variées (éclairage, distance, fond).

| Fichier | Format |
|---|---|
| `dataset/images/` | JPEG, fond clair, pièces non superposées |
| `dataset/annotations.csv` | `filename`, `count`, `total`, `groupe` |

```csv
filename,count,total,grp
1.jpg,8,5.38,grp9
2.jpg,11,8.11,grp9
...
```

---

## Résultats

Évaluation sur les 14 images du dataset :

| Métrique | Valeur |
|---|---|
| Accuracy du comptage | **100 %** |
| Accuracy du montant total | 57,1 % (8/14) |
| MAE nombre de pièces | 0,00 |
| MAE montant total | 0,77 € |

| Image | Nb réel | Nb prédit | Total réel | Total prédit | ✓ |
|---|---|---|---|---|---|
| 1.jpg | 8 | 8 | 5,38 € | 5,38 € | ✅ |
| 2.jpg | 11 | 11 | 8,11 € | 5,53 € | ❌ |
| 3.jpg | 7 | 7 | 7,65 € | 5,20 € | ❌ |
| 4.jpg | 8 | 8 | 3,51 € | 2,01 € | ❌ |
| 5.jpg | 7 | 7 | 3,16 € | 3,16 € | ✅ |
| 6.jpg | 8 | 8 | 5,38 € | 4,37 € | ❌ |
| 7.jpg | 7 | 7 | 7,65 € | 5,20 € | ❌ |
| 8.jpg | 6 | 6 | 2,86 € | 2,86 € | ✅ |
| 9.jpg | 8 | 8 | 3,51 € | 3,51 € | ✅ |
| 10.jpg | 7 | 7 | 3,16 € | 3,16 € | ✅ |
| 11.jpg | 8 | 8 | 2,59 € | 2,59 € | ✅ |
| 12.jpg | 7 | 7 | 4,30 € | 3,50 € | ❌ |
| 13.jpg | 2 | 2 | 3,00 € | 3,00 € | ✅ |
| 14.jpg | 4 | 4 | 7,00 € | 7,00 € | ✅ |

---

## Paramètres

Les principaux paramètres de `detect_coins()` à ajuster selon tes images :

| Paramètre | Défaut | Rôle |
|---|---|---|
| `param2` | `45` | Seuil accumulateur Hough — baisser si trop peu de détections |
| `param1` | `80` | Seuil Canny interne — baisser si les contours sont mal détectés |
| `min_dist_ratio` | `0.15` | Distance min entre centres (fraction de la largeur) |
| `min_r_ratio` | `0.04` | Rayon min des cercles cherchés |
| `max_r_ratio` | `0.22` | Rayon max des cercles cherchés |
| `dp` | `1.2` | Résolution de l'accumulateur (1.0 = plus précis, plus lent) |
