import pandas as pd

from ceres_package.ml_logic.data import load_from_gcp
from ceres_package.ml_logic.ml_preprocess import preprocess_meteo_ml
from ceres_package.params import *

######## RISQUES VERNALISATION #########

def vernalisation_nb_jours_0_10C(df_meteo):
    """
    Identifie les départements présentant un risque de mauvaise vernalisation.

    Calcule, pour chaque département et année, le nombre de jours entre
    décembre et février avec une température moyenne comprise entre 0°C et 10°C.
    Un indicateur binaire vaut 1 lorsque ce nombre est inférieur à 15 jours,
    sinon 0.

    Returns
    -------
    pd.DataFrame
        Colonnes :
        - annee
        - DEPT_ID
        - nb_jours_0_10c
        - vernalisation_nb_jours_0_10C
    """
    df_vernalisation = df_meteo[df_meteo['saison'] == 'vernalisation']

    # Jours avec température comprise entre 0 et 10°C
    mask_temp_0_10c = (
        (df_vernalisation["temp_air_c"] >= 0)
        & (df_vernalisation["temp_air_c"] <= 10)
    )

    df_vern_nb_jours_0_10 = (
        df_vernalisation
        .assign(nb_jours_0_10c=mask_temp_0_10c.astype(int))
        .groupby(["harvest_year", "DEPT_ID"], as_index=False)["nb_jours_0_10c"]
        .sum()
    )

    # 1 = risque de mauvaise vernalisation
    df_vern_nb_jours_0_10["vernalisation_nb_jours_0_10C"] = (
        df_vern_nb_jours_0_10["nb_jours_0_10c"] < 15
    ).astype(int)

    df_vern_nb_jours_0_10 = df_vern_nb_jours_0_10.drop(columns="nb_jours_0_10c")

    return df_vern_nb_jours_0_10

def vernalisation_winter_temperature_anomaly_indicator(df_meteo):
    """
    Calcule un indicateur d'hiver anormalement chaud par département.

    Pour chaque couple (DEPT_ID, annee), calcule la température moyenne hivernale
    (décembre à février), puis compare cette valeur au 90e percentile historique
    des températures hivernales du même département.

    Returns
    -------
    pd.DataFrame
        - DEPT_ID
        - annee
        - hiver_anormalement_chaud (1 si > P90, sinon 0)
    """

    # 1. Chargement des données
    df_vernalisation = df_meteo[df_meteo['saison'] == 'vernalisation']

    # 4. Température moyenne hivernale par département et année
    df_vern_chaleur_anormale = (
        df_vernalisation
        .groupby(["DEPT_ID", "harvest_year"])["temp_air_c"]
        .mean()
        .reset_index(name="temp_moy_hiver")
    )

    # 5. Seuil 90e percentile par département (historique)
    df_vern_chaleur_anormale["seuil_p90_historique"] = (
        df_vern_chaleur_anormale
        .groupby("DEPT_ID")["temp_moy_hiver"]
        .transform(lambda x: x.quantile(0.9))
    )

    # 6. Indicateur binaire
    df_vern_chaleur_anormale["hiver_anormalement_chaud"] = (
        df_vern_chaleur_anormale["temp_moy_hiver"] > df_vern_chaleur_anormale["seuil_p90_historique"]
    ).astype(int)

    df_vern_chaleur_anormale = df_vern_chaleur_anormale.groupby(['DEPT_ID','harvest_year'])['hiver_anormalement_chaud'].sum().reset_index()

    return df_vern_chaleur_anormale

######## RISQUES VEGETATIF #########

def vegetatif_rain_extreme_indicator(df_meteo):
    """
    Calcule un indicateur de pluie extrême basé sur :
    - un cumul glissant de 10 jours
    - un seuil historique (90e percentile par département)

    Retourne un DataFrame enrichi avec :
    - cumul_pluie_10j
    - seuil_p90_pluie_10j
    - pluie_extreme (0/1)
    """

    df_vegetatif = df_meteo[df_meteo['saison'] == 'tallage']

    # --- Calcul du cumul de pluie sur 10 jours glissants ---
    df_vegetatif = df_vegetatif.sort_values(["DEPT_ID", "day"])
    df_vegetatif["cumul_pluie_10j"] = (
        df_vegetatif
        .groupby("DEPT_ID")["precipitations_1h_mm"]
        .rolling(window=10, min_periods=10)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # --- Suppression des fenêtres incomplètes ---
    df_vegetatif_pluie_extreme = df_vegetatif.dropna(subset=["cumul_pluie_10j"])

    # --- Calcul du seuil historique (P90) par département ---
    df_vegetatif_pluie_extreme["seuil_p90_pluie_10j"] = (
        df_vegetatif_pluie_extreme
        .groupby("DEPT_ID")["cumul_pluie_10j"]
        .transform(lambda x: x.quantile(0.9))
    )

    # --- Détection des épisodes de pluie extrême ---
    df_vegetatif_pluie_extreme["pluie_extreme_10j"] = (
        df_vegetatif_pluie_extreme["cumul_pluie_10j"] > df_vegetatif_pluie_extreme["seuil_p90_pluie_10j"]
    ).astype(int)

    df_vegetatif_pluie_extreme = df_vegetatif_pluie_extreme.groupby(['DEPT_ID','harvest_year'])['pluie_extreme_10j'].sum().reset_index()

    return df_vegetatif_pluie_extreme

def vegetatif_humid_high_indicator(df_meteo):
    """
    Détecte les épisodes d'humidité persistante (>85%) sur la période végétative
    (tallage) lorsque la durée consécutive est >= 5 jours.

    Retourne le nombre d'épisodes par département et année.
    """

    df = df_meteo[df_meteo["saison"] == "tallage"].copy()

    df = df[
        ["DEPT_ID", "harvest_year", "day", "humidite_relative_pct"]
    ].copy()

    # IMPORTANT : tri temporel
    df = df.sort_values(["DEPT_ID", "harvest_year", "day"])

    # signal binaire
    df["humid_high"] = (df["humidite_relative_pct"] > 85).astype(int)

    # run id (OK)
    df["humid_run_id"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["humid_high"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    # longueur des runs
    run_lengths = (
        df.groupby(["DEPT_ID", "harvest_year", "humid_run_id"])["humid_high"]
        .sum()
        .rename("run_length")
        .reset_index()
    )

    df = df.merge(run_lengths, on=["DEPT_ID", "harvest_year", "humid_run_id"])

    # épisodes valides
    episodes = df[(df["humid_high"] == 1) & (df["run_length"] >= 5)]

    # nombre d'épisodes
    df_vegetatif_humidite_elevee = (
        episodes
        .groupby(["DEPT_ID", "harvest_year"])
        .size()
        .reset_index(name="humidity_persistence_risk")
    )

    # compléter les zéros
    spine = df[["DEPT_ID", "harvest_year"]].drop_duplicates()

    df_vegetatif_humidite_elevee = (
        spine.merge(df_vegetatif_humidite_elevee, on=["DEPT_ID", "harvest_year"], how="left")
             .fillna({"humidity_persistence_risk": 0})
             .astype({"humidity_persistence_risk": int})
    )

    return df_vegetatif_humidite_elevee

def vegetatif_rayonnement_faible_indicator(df_meteo):
    """
    Retourne un DataFrame contenant le nombre d'épisodes de
    rayonnement faible (>= 10 jours consécutifs sous le P10 historique)
    par année et par département.
    """
    # filtre saison
    df = df_meteo[df_meteo["saison"] == "tallage"]

    df = df[
        ["DEPT_ID", "day", "harvest_year", "rayonnement_global_jcm2"]
    ].copy()

    # tri temporel obligatoire
    df = df.sort_values(["DEPT_ID", "harvest_year", "day"])

    # seuil P10 par département
    df["seuil_p10_rayonnement"] = (
        df.groupby("DEPT_ID")["rayonnement_global_jcm2"]
        .transform(lambda x: x.quantile(0.10))
    )

    # signal binaire
    df["rayonnement_faible"] = (
        df["rayonnement_global_jcm2"] < df["seuil_p10_rayonnement"]
    ).astype(int)

    # run id (IMPORTANT: inclure harvest_year)
    df["rayonnement_run_id"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["rayonnement_faible"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    # longueur des runs
    run_lengths = (
        df.groupby(["DEPT_ID", "harvest_year", "rayonnement_run_id"])["rayonnement_faible"]
        .sum()
        .reset_index(name="run_length")
    )

    df = df.merge(run_lengths, on=["DEPT_ID", "harvest_year", "rayonnement_run_id"])

    # épisodes valides (>=10 jours consécutifs)
    episodes = df[(df["rayonnement_faible"] == 1) & (df["run_length"] >= 10)]

    # compter les épisodes
    out = (
        episodes
        .groupby(["DEPT_ID", "harvest_year"])
        .size()
        .reset_index(name="nb_risques_rayonnement_faible")
    )

    # compléter les zéros
    spine = df[["DEPT_ID", "harvest_year"]].drop_duplicates()

    df_vegetatif_rayonnement_faible = (
        spine.merge(out, on=["DEPT_ID", "harvest_year"], how="left")
             .fillna({"nb_risques_rayonnement_faible": 0})
             .astype({"nb_risques_rayonnement_faible": int})
    )

    return df_vegetatif_rayonnement_faible

######## RISQUES FLORAISON #########

def floraison_nb_jours_tx_gt_25(df_meteo):
    """
    Nombre de jours avec TX > 25°C pendant la floraison.

    Returns
    -------
    pd.DataFrame
        DEPT_ID, harvest_year, floraison_nb_jours_tx_gt_25
    """
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()

    df["flag"] = (df["temp_max_c"] > 25).astype(int)

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["flag"]
        .sum()
        .rename(columns={"flag": "floraison_nb_jours_tx_gt_25"})
    )

def floraison_nb_jours_tx_gt_30(df_meteo):
    """
    Nombre de jours avec TX > 30°C pendant la floraison.
    """
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()

    df["flag"] = (df["temp_max_c"] > 30).astype(int)

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["flag"]
        .sum()
        .rename(columns={"flag": "floraison_nb_jours_tx_gt_30"})
    )

def floraison_cumul_pluie_10j_max(df_meteo):
    """
    Maximum du cumul de pluie glissant sur 10 jours pendant la floraison.
    """
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "day"])

    df["cumul_pluie_10j"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["precipitations_1h_mm"]
        .rolling(window=10, min_periods=10)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    return (
        df.dropna(subset=["cumul_pluie_10j"])
        .groupby(["DEPT_ID", "harvest_year"], as_index=False)["cumul_pluie_10j"]
        .max()
        .rename(columns={"cumul_pluie_10j": "floraison_cumul_pluie_10j_max"})
    )

def floraison_max_consecutive_tx_gt_25(df_meteo):
    """
    Plus longue séquence de jours consécutifs avec TX > 25°C (floraison).
    """
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "day"])

    df["flag"] = (df["temp_max_c"] > 25).astype(int)

    df["run_id"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["flag"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    run_lengths = (
        df.groupby(["DEPT_ID", "harvest_year", "run_id"])["flag"]
        .sum()
        .reset_index(name="run_length")
    )

    df = df.merge(run_lengths, on=["DEPT_ID", "harvest_year", "run_id"])

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["run_length"]
        .max()
        .rename(columns={"run_length": "floraison_max_consecutive_tx_gt_25"})
    )

def floraison_max_consecutive_tx_gt_30(df_meteo):
    """
    Plus longue séquence de jours consécutifs avec TX > 30°C (floraison).
    """
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "day"])

    df["flag"] = (df["temp_max_c"] > 30).astype(int)

    df["run_id"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["flag"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    run_lengths = (
        df.groupby(["DEPT_ID", "harvest_year", "run_id"])["flag"]
        .sum()
        .reset_index(name="run_length")
    )

    df = df.merge(run_lengths, on=["DEPT_ID", "harvest_year", "run_id"])

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["run_length"]
        .max()
        .rename(columns={"run_length": "floraison_max_consecutive_tx_gt_30"})
    )

def floraison_heat_degree_days_25(df_meteo):
    """
    Somme des degrés-jours de chaleur : Σ(TX - 25) pour TX > 25°C.
    """
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()

    df["heat_dd"] = (df["temp_max_c"] - 25).clip(lower=0)

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["heat_dd"]
        .sum()
        .rename(columns={"heat_dd": "floraison_heat_degree_days_25"})
    )

def floraison_heat_degree_days_30(df_meteo):
    """
    Somme des degrés-jours de chaleur sévère : Σ(TX - 30) pour TX > 30°C.
    """
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()

    df["heat_dd"] = (df["temp_max_c"] - 30).clip(lower=0)

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["heat_dd"]
        .sum()
        .rename(columns={"heat_dd": "floraison_heat_degree_days_30"})
    )

def floraison_episodes_tx_gt_30_ge3(df_meteo):
    """
    Nombre d'épisodes de chaleur sévère : TX > 30°C pendant >= 3 jours consécutifs.
    """
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "day"])

    df["flag"] = (df["temp_max_c"] > 30).astype(int)

    df["run_id"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["flag"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    run_lengths = (
        df.groupby(["DEPT_ID", "harvest_year", "run_id"])["flag"]
        .sum()
        .reset_index(name="run_length")
    )

    df = df.merge(run_lengths, on=["DEPT_ID", "harvest_year", "run_id"])

    episodes = df[(df["flag"] == 1) & (df["run_length"] >= 3)]

    out = (
        episodes
        .groupby(["DEPT_ID", "harvest_year"])
        .size()
        .reset_index(name="floraison_episodes_tx_gt_30_ge3")
    )

    spine = df[["DEPT_ID", "harvest_year"]].drop_duplicates()

    return (
        spine.merge(out, on=["DEPT_ID", "harvest_year"], how="left")
        .fillna({"floraison_episodes_tx_gt_30_ge3": 0})
        .astype({"floraison_episodes_tx_gt_30_ge3": int})
    )

def floraison_days_heat_drought(df_meteo):
    """
    Nombre de jours avec TX > 28°C et pluie journalière < P10 (par dept×harvest_year).
    """
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()

    df["seuil_p10_pluie"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["precipitations_1h_mm"]
        .transform(lambda x: x.quantile(0.10))
    )

    df["flag"] = (
        (df["temp_max_c"] > 28)
        & (df["precipitations_1h_mm"] < df["seuil_p10_pluie"])
    ).astype(int)

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["flag"]
        .sum()
        .rename(columns={"flag": "floraison_days_heat_drought"})
    )

######## RISQUES REMPLISSAGE #########

def remplissage_nb_jours_tx_gt_25(df_meteo):
    """
    Nombre de jours avec TX > 25°C pendant le remplissage du grain.
    """
    df = df_meteo[df_meteo["saison"] == "remplissage"].copy()

    df["flag"] = (df["temp_max_c"] > 25).astype(int)

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["flag"]
        .sum()
        .rename(columns={"flag": "remplissage_nb_jours_tx_gt_25"})
    )

def remplissage_heatwave_length_max(df_meteo):
    """
    Longueur maximale d'une vague de chaleur : séquence consécutive avec TX > 28°C.
    """
    df = df_meteo[df_meteo["saison"] == "remplissage"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "day"])

    df["flag"] = (df["temp_max_c"] > 28).astype(int)

    df["run_id"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["flag"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    run_lengths = (
        df.groupby(["DEPT_ID", "harvest_year", "run_id"])["flag"]
        .sum()
        .reset_index(name="run_length")
    )

    df = df.merge(run_lengths, on=["DEPT_ID", "harvest_year", "run_id"])

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["run_length"]
        .max()
        .rename(columns={"run_length": "remplissage_heatwave_length_max"})
    )

def remplissage_episodes_heatwave_ge3(df_meteo):
    """
    Nombre d'épisodes de vague de chaleur : TX > 28°C pendant >= 3 jours consécutifs.
    """
    df = df_meteo[df_meteo["saison"] == "remplissage"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "day"])

    df["flag"] = (df["temp_max_c"] > 28).astype(int)

    df["run_id"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["flag"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    run_lengths = (
        df.groupby(["DEPT_ID", "harvest_year", "run_id"])["flag"]
        .sum()
        .reset_index(name="run_length")
    )

    df = df.merge(run_lengths, on=["DEPT_ID", "harvest_year", "run_id"])

    episodes = df[(df["flag"] == 1) & (df["run_length"] >= 3)]

    out = (
        episodes
        .groupby(["DEPT_ID", "harvest_year"])
        .size()
        .reset_index(name="remplissage_episodes_heatwave_ge3")
    )

    spine = df[["DEPT_ID", "harvest_year"]].drop_duplicates()

    return (
        spine.merge(out, on=["DEPT_ID", "harvest_year"], how="left")
        .fillna({"remplissage_episodes_heatwave_ge3": 0})
        .astype({"remplissage_episodes_heatwave_ge3": int})
    )

def remplissage_max_consecutive_dry_days(df_meteo):
    """
    Plus longue séquence de jours consécutifs sans pluie (remplissage).
    """
    df = df_meteo[df_meteo["saison"] == "remplissage"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "day"])

    df["flag"] = (df["precipitations_1h_mm"] <= 0).astype(int)

    df["run_id"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["flag"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    run_lengths = (
        df.groupby(["DEPT_ID", "harvest_year", "run_id"])["flag"]
        .sum()
        .reset_index(name="run_length")
    )

    df = df.merge(run_lengths, on=["DEPT_ID", "harvest_year", "run_id"])

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["run_length"]
        .max()
        .rename(columns={"run_length": "remplissage_max_consecutive_dry_days"})
    )

def remplissage_storm_event_count(df_meteo):
    """
    Nombre de jours à risque de verse : pluie > 15 mm et rafale > 15 m/s.
    """
    df = df_meteo[df_meteo["saison"] == "remplissage"].copy()

    df["flag"] = (
        (df["precipitations_1h_mm"] > 15)
        & (df["rafale_max_ms"] > 15)
    ).astype(int)

    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["flag"]
        .sum()
        .rename(columns={"flag": "remplissage_storm_event_count"})
    )


######### DF GLOBAL FEATURES ############

JOIN_KEYS = ["DEPT_ID", "harvest_year"]

VERN_FUNCTIONS = [
    vernalisation_nb_jours_0_10C,
    vernalisation_winter_temperature_anomaly_indicator,
]

VEG_FUNCTIONS = [
    vegetatif_rain_extreme_indicator,
    vegetatif_humid_high_indicator,
    vegetatif_rayonnement_faible_indicator,
]

FLOR_FUNCTIONS = [
    floraison_nb_jours_tx_gt_25,
    floraison_nb_jours_tx_gt_30,
    floraison_cumul_pluie_10j_max,
    floraison_max_consecutive_tx_gt_25,
    floraison_max_consecutive_tx_gt_30,
    floraison_heat_degree_days_25,
    floraison_heat_degree_days_30,
    floraison_episodes_tx_gt_30_ge3,
    floraison_days_heat_drought,
]

REMP_FUNCTIONS = [
    remplissage_nb_jours_tx_gt_25,
    remplissage_heatwave_length_max,
    remplissage_episodes_heatwave_ge3,
    remplissage_max_consecutive_dry_days,
    remplissage_storm_event_count,
]

ENGINEERED_FEATURE_FUNCTIONS = (
    VERN_FUNCTIONS + VEG_FUNCTIONS + FLOR_FUNCTIONS + REMP_FUNCTIONS
)

# Sub-combos : concat de listes par stade — ajouter / retirer une ligne pour expérimenter
FEATURE_COMBOS = {
    "vern_full": VERN_FUNCTIONS,
    "veg_full": VEG_FUNCTIONS,
    "flor_full": FLOR_FUNCTIONS,
    "remp_full": REMP_FUNCTIONS,
    "early_stages": VERN_FUNCTIONS + VEG_FUNCTIONS,
    "flor_remp": FLOR_FUNCTIONS + REMP_FUNCTIONS,
    "all": ENGINEERED_FEATURE_FUNCTIONS,
}

def build_engineered_features_mart(df_meteo, combo_id="all"):
    """
    Assemble les features agronomiques en un seul DataFrame.

    Parameters
    ----------
    df_meteo : pd.DataFrame
        Sortie de preprocess_meteo_ml().
    combo_id : str
        Clé dans FEATURE_COMBOS (défaut "all" = toutes les features).

    Returns
    -------
    pd.DataFrame
        DEPT_ID, harvest_year + colonnes du combo choisi.
    """
    if combo_id not in FEATURE_COMBOS:
        raise ValueError(
            f"combo_id inconnu : {combo_id!r}. "
            f"Disponibles : {list(FEATURE_COMBOS.keys())}"
        )

    df_mart = None

    for feature_fn in FEATURE_COMBOS[combo_id]:
        df_feat = feature_fn(df_meteo)
        feat_cols = [c for c in df_feat.columns if c not in JOIN_KEYS]
        df_feat = df_feat[JOIN_KEYS + feat_cols]

        if df_mart is None:
            df_mart = df_feat
        else:
            df_mart = df_mart.merge(df_feat, on=JOIN_KEYS, how="outer")

    return df_mart.sort_values(JOIN_KEYS).reset_index(drop=True)
