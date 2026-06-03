# Données Climatologiques Horaires — Météo-France

> Source : [data.gouv.fr — Données climatologiques de base horaires](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-horaires)
> Licence : Licence Ouverte / Open Licence version 2.0

---

## Description

Dataset des observations météorologiques horaires des stations Météo-France couvrant la France métropolitaine (96 départements), de 2010 à 2026.

Les données sont issues du réseau de stations automatiques et synoptiques de Météo-France. Chaque ligne correspond à une mesure horaire pour une station donnée.

---

## Structure des fichiers

```
raw_data/meteofrance/
├── dept_01.csv                             # Données filtrées et nettoyées — département 01
├── dept_02.csv
├── ...
├── dept_95.csv
├── .done_01                                # Marqueurs de complétion (reprise automatique)
└── download.log                            # Log des téléchargements
└── meteofrance_horaire_2010_2026.csv
```

---

## Périodes disponibles

| Fichier | Période couverte | Statut |
|---|---|---|
| `H_{DEPT}_2010-2019.csv.gz` | 2010 → 2019 | Archivé, stable |
| `H_{DEPT}_previous-2020-2024.csv.gz` | 2020 → 2024 | Archivé, stable |
| `H_{DEPT}_latest-2025-2026.csv.gz` | 2025 → aujourd'hui | Mis à jour quotidiennement |

---

## Colonnes sélectionnées

Parmi les 206 colonnes du dataset original, 37 colonnes ont été retenues pour leur pertinence agronomique (étude blé tendre).

### Identifiants & localisation

| Colonne originale | Colonne renommée | Description | Unité |
|---|---|---|---|
| `NUM_POSTE` | `id_station` | Identifiant Météo-France de la station | 8 chiffres |
| `LAT` | `latitude` | Latitude de la station | degrés décimaux |
| `LON` | `longitude` | Longitude de la station | degrés décimaux |
| `ALTI` | `altitude_m` | Altitude du pied de l'abri | m |
| `AAAAMMJJHH` | `datetime` | Date et heure de la mesure | YYYYMMDDHH |

### Température

| Colonne originale | Colonne renommée | Description | Unité |
|---|---|---|---|
| `T` | `temp_air_c` | Température sous abri instantanée | °C (÷10) |
| `TN` | `temp_min_c` | Température minimale dans l'heure | °C (÷10) |
| `TX` | `temp_max_c` | Température maximale dans l'heure | °C (÷10) |
| `TD` | `temp_rosee_c` | Température du point de rosée | °C (÷10) |
| `TNSOL` | `temp_min_sol_10cm_c` | Température minimale à 10 cm au-dessus du sol | °C (÷10) |
| `TCHAUSSEE` | `temp_surface_sol_c` | Température de surface (herbe ou bitume) | °C (÷10) |
| `T10` | `temp_sol_10cm_c` | Température sol à -10 cm | °C (÷10) |
| `T20` | `temp_sol_20cm_c` | Température sol à -20 cm | °C (÷10) |
| `T50` | `temp_sol_50cm_c` | Température sol à -50 cm | °C (÷10) |
| `T100` | `temp_sol_100cm_c` | Température sol à -1 m | °C (÷10) |
| `DG` | `duree_gel_min` | Durée de gel (T ≤ 0°C) dans l'heure | minutes |

### Humidité

| Colonne originale | Colonne renommée | Description | Unité |
|---|---|---|---|
| `U` | `humidite_relative_pct` | Humidité relative | % |
| `UN` | `humidite_min_pct` | Humidité relative minimale dans l'heure | % |
| `HUN` | `heure_humidite_min` | Heure de l'humidité minimale | hhmm |
| `UX` | `humidite_max_pct` | Humidité relative maximale dans l'heure | % |
| `HUX` | `heure_humidite_max` | Heure de l'humidité maximale | hhmm |
| `DHUMI40` | `duree_humidite_inf40_min` | Durée avec humidité ≤ 40% | minutes |
| `DHUMI80` | `duree_humidite_sup80_min` | Durée avec humidité ≥ 80% | minutes |
| `DHUMEC` | `duree_humectation_foliaire_min` | Durée d'humectation foliaire | minutes |
| `TSV` | `tension_vapeur_hpa` | Tension de vapeur | hPa (÷10) |

### Précipitations

| Colonne originale | Colonne renommée | Description | Unité |
|---|---|---|---|
| `RR1` | `precipitations_1h_mm` | Précipitations en 1 heure | mm (÷10) |
| `DRR1` | `duree_precipitations_min` | Durée des précipitations | minutes |

### Vent

| Colonne originale | Colonne renommée | Description | Unité |
|---|---|---|---|
| `FF` | `vent_moyen_10m_ms` | Vent moyen sur 10 min à 10 m | m/s (÷10) |
| `FXI` | `rafale_max_ms` | Rafale maximale instantanée dans l'heure | m/s (÷10) |
| `DD` | `direction_vent_deg` | Direction du vent | degrés (rose 360°) |

### Rayonnement & ensoleillement

| Colonne originale | Colonne renommée | Description | Unité |
|---|---|---|---|
| `GLO` | `rayonnement_global_jcm2` | Rayonnement global horaire (UTC) | J/cm² |
| `INS` | `insolation_min` | Insolation horaire (UTC) | minutes |

### Neige & sol

| Colonne originale | Colonne renommée | Description | Unité |
|---|---|---|---|
| `NEIGETOT` | `hauteur_neige_sol_cm` | Hauteur totale de neige au sol | cm |
| `HNEIGEF` | `neige_fraiche_6h_cm` | Neige fraîche tombée en 6h | cm |
| `SOL` | `etat_sol` | État du sol sans neige (code OMM 0–9) | code |

### Pression

| Colonne originale | Colonne renommée | Description | Unité |
|---|---|---|---|
| `PMER` | `pression_mer_hpa` | Pression au niveau de la mer | hPa (÷10) |

---

## Codes qualité

Chaque colonne de mesure est associée dans le dataset brut à une colonne qualité préfixée `Q` (ex : `QT` pour `T`).

| Code | Signification |
|---|---|
| `0` | Donnée **protégée** — validée définitivement par un climatologue |
| `1` | Donnée **validée** — contrôle automatique ou climatologue |
| `9` | Donnée **filtrée** — a passé les contrôles de premier niveau |
| `2` | Donnée **douteuse** — en cours de vérification, à exclure |

> Les colonnes qualité ont été supprimées après vérification : aucune donnée douteuse (code `2`) n'a été trouvée dans le dataset.

---

## Colonnes exclues et justification

### Entièrement vides (0 valeurs)

| Groupe | Colonnes | Raison |
|---|---|---|
| Vent à 2 m | `FF2`, `DD2`, `FXI2`, `DXI2`, `HXI2` | Stations spécialisées uniquement |
| Géopotentiel | `GEOP` | Stations d'altitude > 750 m uniquement |
| Mer | `TMER`, `DIRHOULE`, `HVAGUE`, `PVAGUE` | Sémaphores côtiers uniquement |
| Neige avancée | `TSNEIGE`, `TUBENEIGE`, `ESNEIGE` | Stations nivométéo uniquement |
| Rayonnement détaillé | `DIR`, `DIF`, `UV`, `INFRAR` + versions TSV | Stations équipées uniquement |
| Inutilisés | `TLAGON`, `TVEGETAUX`, `ECOULEMENT` | Documentés comme inutilisés |

### Plus de 80% de valeurs manquantes

| Groupe | Colonnes | Raison |
|---|---|---|
| Nuages détaillés | `CL`, `CM`, `CH`, `N1`→`N4`, `C1`→`C4`, `B1`→`B4` | Observations manuelles synoptiques (~150 stations) |
| Visibilité & temps présent | `VV`, `DVV200`, `WW`, `W1`, `W2` | Observations manuelles ou capteurs rares |
| État du sol avec neige | `SOLNG` | Observations manuelles synoptiques |
| Neige spécialisée | `HNEIGEFI3`, `HNEIGEFI1`, `CHARGENEIGE` | Stations nivométéo uniquement |

---

## Codes état du sol (`etat_sol`)

| Code | Description |
|---|---|
| 0 | Surface sèche |
| 1 | Surface humide |
| 2 | Surface mouillée (eau stagnante) |
| 3 | Inondé |
| 4 | Surface gelée |
| 5 | Verglas |
| 6 | Poussière ou sable meuble ne couvrant pas complètement le sol |
| 7 | Couche fine de poussière ou sable couvrant complètement le sol |
| 8 | Couche épaisse de poussière ou sable couvrant complètement le sol |
| 9 | Très sec avec fissures |

---

## Notes techniques

- Les valeurs de température (`T`, `TN`, `TX`, `TD`, `TNSOL`, `TCHAUSSEE`, `T10`, `T20`, `T50`, `T100`) sont stockées en **°C × 10** dans les fichiers bruts. Diviser par 10 pour obtenir des °C.
- Les valeurs de vent (`FF`, `FXI`) sont en **m/s × 10**. Diviser par 10 pour obtenir des m/s.
- Les valeurs de pression (`PMER`, `TSV`) sont en **hPa × 10**. Diviser par 10 pour obtenir des hPa.
- Les précipitations (`RR1`) sont en **mm × 10**. Diviser par 10 pour obtenir des mm.
- La colonne `datetime` est au format `YYYYMMDDHH` dans les fichiers bruts — à convertir avec `pd.to_datetime(..., format="%Y%m%d%H")`.
- `PMER` n'est renseignée que pour les stations dont l'altitude est ≤ 750 m.
