import pandas as pd
import numpy as np
from ceres_package.params import *


######## RISQUES VERNALISATION — DL (hourly) #########

def vernalisation_nb_jours_0_10C_dl(df_meteo):
    """
    Nombre d'heures avec temp entre 0-10°C / 24 = équivalent jours.
    Risque si < 15 jours équivalents.
    """
    df_vernalisation = df_meteo[df_meteo['saison'] == 'vernalisation']

    mask_temp_0_10c = (
        (df_vernalisation["temp_air_c"] >= 0)
        & (df_vernalisation["temp_air_c"] <= 10)
    )

    df_vern = (
        df_vernalisation
        .assign(nb_heures_0_10c=mask_temp_0_10c.astype(int))
        .groupby(["harvest_year", "DEPT_ID"], as_index=False)["nb_heures_0_10c"]
        .sum()
    )

    df_vern["vernalisation_nb_jours_0_10C"] = (
        (df_vern["nb_heures_0_10c"] / 24) < 15
    ).astype(int)

    return df_vern.drop(columns="nb_heures_0_10c")


def vernalisation_winter_temperature_anomaly_indicator_dl(df_meteo):
    """
    Indicateur d'hiver anormalement chaud — logique identique, fonctionne sur hourly.
    """
    df_vernalisation = df_meteo[df_meteo['saison'] == 'vernalisation']

    df_vern = (
        df_vernalisation
        .groupby(["DEPT_ID", "harvest_year"])["temp_air_c"]
        .mean()
        .reset_index(name="temp_moy_hiver")
    )

    df_vern["seuil_p90_historique"] = (
        df_vern.groupby("DEPT_ID")["temp_moy_hiver"]
        .transform(lambda x: x.quantile(0.9))
    )

    df_vern["hiver_anormalement_chaud"] = (
        df_vern["temp_moy_hiver"] > df_vern["seuil_p90_historique"]
    ).astype(int)

    return df_vern.groupby(['DEPT_ID', 'harvest_year'])['hiver_anormalement_chaud'].sum().reset_index()


######## RISQUES VEGETATIF — DL (hourly) #########

def vegetatif_rain_extreme_indicator_dl(df_meteo):
    """
    Cumul glissant sur 240h (10 jours x 24h) vs P90 historique par département.
    """
    df_vegetatif = df_meteo[df_meteo['saison'] == 'tallage'].copy()
    df_vegetatif = df_vegetatif.sort_values(["DEPT_ID", "datetime"])

    df_vegetatif["cumul_pluie_240h"] = (
        df_vegetatif
        .groupby("DEPT_ID")["precipitations_1h_mm"]
        .rolling(window=240, min_periods=240)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df_veg = df_vegetatif.dropna(subset=["cumul_pluie_240h"])

    df_veg["seuil_p90"] = (
        df_veg.groupby("DEPT_ID")["cumul_pluie_240h"]
        .transform(lambda x: x.quantile(0.9))
    )

    df_veg["pluie_extreme_10j"] = (
        df_veg["cumul_pluie_240h"] > df_veg["seuil_p90"]
    ).astype(int)

    return df_veg.groupby(['DEPT_ID', 'harvest_year'])['pluie_extreme_10j'].sum().reset_index()


def vegetatif_humid_high_indicator_dl(df_meteo):
    """
    Épisodes d'humidité > 85% pendant >= 120h consécutives (5 jours × 24h).
    """
    df = df_meteo[df_meteo["saison"] == "tallage"].copy()
    df = df[["DEPT_ID", "harvest_year", "datetime", "humidite_relative_pct"]].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "datetime"])

    df["humid_high"] = (df["humidite_relative_pct"] > 85).astype(int)

    df["humid_run_id"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["humid_high"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    run_lengths = (
        df.groupby(["DEPT_ID", "harvest_year", "humid_run_id"])["humid_high"]
        .sum()
        .rename("run_length")
        .reset_index()
    )

    df = df.merge(run_lengths, on=["DEPT_ID", "harvest_year", "humid_run_id"])

    episodes = df[(df["humid_high"] == 1) & (df["run_length"] >= 120)]

    df_out = (
        episodes
        .groupby(["DEPT_ID", "harvest_year"])
        .size()
        .reset_index(name="humidity_persistence_risk")
    )

    spine = df[["DEPT_ID", "harvest_year"]].drop_duplicates()
    return (
        spine.merge(df_out, on=["DEPT_ID", "harvest_year"], how="left")
             .fillna({"humidity_persistence_risk": 0})
             .astype({"humidity_persistence_risk": int})
    )


def vegetatif_rayonnement_faible_indicator_dl(df_meteo):
    """
    Épisodes de rayonnement faible >= 240h consécutives (10 jours × 24h) sous P10.
    """
    df = df_meteo[df_meteo["saison"] == "tallage"].copy()
    df = df[["DEPT_ID", "datetime", "harvest_year", "rayonnement_global_jcm2"]].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "datetime"])

    df["seuil_p10"] = (
        df.groupby("DEPT_ID")["rayonnement_global_jcm2"]
        .transform(lambda x: x.quantile(0.10))
    )

    df["rayonnement_faible"] = (
        df["rayonnement_global_jcm2"] < df["seuil_p10"]
    ).astype(int)

    df["rayonnement_run_id"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["rayonnement_faible"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    run_lengths = (
        df.groupby(["DEPT_ID", "harvest_year", "rayonnement_run_id"])["rayonnement_faible"]
        .sum()
        .reset_index(name="run_length")
    )

    df = df.merge(run_lengths, on=["DEPT_ID", "harvest_year", "rayonnement_run_id"])

    episodes = df[(df["rayonnement_faible"] == 1) & (df["run_length"] >= 240)]

    out = (
        episodes
        .groupby(["DEPT_ID", "harvest_year"])
        .size()
        .reset_index(name="nb_risques_rayonnement_faible")
    )

    spine = df[["DEPT_ID", "harvest_year"]].drop_duplicates()
    return (
        spine.merge(out, on=["DEPT_ID", "harvest_year"], how="left")
             .fillna({"nb_risques_rayonnement_faible": 0})
             .astype({"nb_risques_rayonnement_faible": int})
    )


######## RISQUES FLORAISON — DL (hourly) #########

def floraison_nb_jours_tx_gt_25_dl(df_meteo):
    """Nb heures avec temp_air_c > 25°C / 24 = équivalent jours."""
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df["flag"] = (df["temp_air_c"] > 25).astype(int)
    df_out = (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["flag"]
        .sum()
        .rename(columns={"flag": "floraison_nb_jours_tx_gt_25"})
    )
    df_out["floraison_nb_jours_tx_gt_25"] = df_out["floraison_nb_jours_tx_gt_25"] / 24
    return df_out


def floraison_nb_jours_tx_gt_30_dl(df_meteo):
    """Nb heures avec temp_air_c > 30°C / 24 = équivalent jours."""
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df["flag"] = (df["temp_air_c"] > 30).astype(int)
    df_out = (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["flag"]
        .sum()
        .rename(columns={"flag": "floraison_nb_jours_tx_gt_30"})
    )
    df_out["floraison_nb_jours_tx_gt_30"] = df_out["floraison_nb_jours_tx_gt_30"] / 24
    return df_out


def floraison_cumul_pluie_10j_max_dl(df_meteo):
    """Max du cumul de pluie glissant sur 240h (10j × 24h)."""
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "datetime"])

    df["cumul_pluie_240h"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["precipitations_1h_mm"]
        .rolling(window=240, min_periods=240)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    return (
        df.dropna(subset=["cumul_pluie_240h"])
        .groupby(["DEPT_ID", "harvest_year"], as_index=False)["cumul_pluie_240h"]
        .max()
        .rename(columns={"cumul_pluie_240h": "floraison_cumul_pluie_10j_max"})
    )


def floraison_max_consecutive_tx_gt_25_dl(df_meteo):
    """Plus longue séquence horaire avec temp > 25°C / 24 = équivalent jours."""
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "datetime"])

    df["flag"] = (df["temp_air_c"] > 25).astype(int)
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
    df_out = (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["run_length"]
        .max()
        .rename(columns={"run_length": "floraison_max_consecutive_tx_gt_25"})
    )
    df_out["floraison_max_consecutive_tx_gt_25"] = df_out["floraison_max_consecutive_tx_gt_25"] / 24
    return df_out


def floraison_max_consecutive_tx_gt_30_dl(df_meteo):
    """Plus longue séquence horaire avec temp > 30°C / 24 = équivalent jours."""
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "datetime"])

    df["flag"] = (df["temp_air_c"] > 30).astype(int)
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
    df_out = (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["run_length"]
        .max()
        .rename(columns={"run_length": "floraison_max_consecutive_tx_gt_30"})
    )
    df_out["floraison_max_consecutive_tx_gt_30"] = df_out["floraison_max_consecutive_tx_gt_30"] / 24
    return df_out


def floraison_heat_degree_days_25_dl(df_meteo):
    """Σ(temp_air_c - 25) / 24 pour temp > 25°C."""
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df["heat_dd"] = (df["temp_air_c"] - 25).clip(lower=0) / 24
    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["heat_dd"]
        .sum()
        .rename(columns={"heat_dd": "floraison_heat_degree_days_25"})
    )


def floraison_heat_degree_days_30_dl(df_meteo):
    """Σ(temp_air_c - 30) / 24 pour temp > 30°C."""
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df["heat_dd"] = (df["temp_air_c"] - 30).clip(lower=0) / 24
    return (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["heat_dd"]
        .sum()
        .rename(columns={"heat_dd": "floraison_heat_degree_days_30"})
    )


def floraison_episodes_tx_gt_30_ge3_dl(df_meteo):
    """Épisodes de chaleur sévère : temp > 30°C pendant >= 72h (3j × 24h)."""
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "datetime"])

    df["flag"] = (df["temp_air_c"] > 30).astype(int)
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
    episodes = df[(df["flag"] == 1) & (df["run_length"] >= 72)]

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


def floraison_days_heat_drought_dl(df_meteo):
    """Nb heures avec temp > 28°C et pluie < P10 / 24 = équivalent jours."""
    df = df_meteo[df_meteo["saison"] == "floraison"].copy()

    df["seuil_p10_pluie"] = (
        df.groupby(["DEPT_ID", "harvest_year"])["precipitations_1h_mm"]
        .transform(lambda x: x.quantile(0.10))
    )

    df["flag"] = (
        (df["temp_air_c"] > 28)
        & (df["precipitations_1h_mm"] < df["seuil_p10_pluie"])
    ).astype(int)

    df_out = (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["flag"]
        .sum()
        .rename(columns={"flag": "floraison_days_heat_drought"})
    )
    df_out["floraison_days_heat_drought"] = df_out["floraison_days_heat_drought"] / 24
    return df_out


######## RISQUES REMPLISSAGE — DL (hourly) #########

def remplissage_nb_jours_tx_gt_25_dl(df_meteo):
    """Nb heures avec temp > 25°C / 24 = équivalent jours."""
    df = df_meteo[df_meteo["saison"] == "remplissage"].copy()
    df["flag"] = (df["temp_air_c"] > 25).astype(int)
    df_out = (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["flag"]
        .sum()
        .rename(columns={"flag": "remplissage_nb_jours_tx_gt_25"})
    )
    df_out["remplissage_nb_jours_tx_gt_25"] = df_out["remplissage_nb_jours_tx_gt_25"] / 24
    return df_out


def remplissage_heatwave_length_max_dl(df_meteo):
    """Longueur max vague de chaleur en heures / 24 = équivalent jours."""
    df = df_meteo[df_meteo["saison"] == "remplissage"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "datetime"])

    df["flag"] = (df["temp_air_c"] > 28).astype(int)
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
    df_out = (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["run_length"]
        .max()
        .rename(columns={"run_length": "remplissage_heatwave_length_max"})
    )
    df_out["remplissage_heatwave_length_max"] = df_out["remplissage_heatwave_length_max"] / 24
    return df_out


def remplissage_episodes_heatwave_ge3_dl(df_meteo):
    """Épisodes vague de chaleur : temp > 28°C pendant >= 72h (3j × 24h)."""
    df = df_meteo[df_meteo["saison"] == "remplissage"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "datetime"])

    df["flag"] = (df["temp_air_c"] > 28).astype(int)
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
    episodes = df[(df["flag"] == 1) & (df["run_length"] >= 72)]

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


def remplissage_max_consecutive_dry_days_dl(df_meteo):
    """Plus longue séquence horaire sans pluie / 24 = équivalent jours."""
    df = df_meteo[df_meteo["saison"] == "remplissage"].copy()
    df = df.sort_values(["DEPT_ID", "harvest_year", "datetime"])

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
    df_out = (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["run_length"]
        .max()
        .rename(columns={"run_length": "remplissage_max_consecutive_dry_days"})
    )
    df_out["remplissage_max_consecutive_dry_days"] = df_out["remplissage_max_consecutive_dry_days"] / 24
    return df_out


def remplissage_storm_event_count_dl(df_meteo):
    """Nb heures avec pluie > 15mm et rafale > 15m/s / 24 = équivalent jours."""
    df = df_meteo[df_meteo["saison"] == "remplissage"].copy()

    df["flag"] = (
        (df["precipitations_1h_mm"] > 15)
        & (df["rafale_max_ms"] > 15)
    ).astype(int)

    df_out = (
        df.groupby(["DEPT_ID", "harvest_year"], as_index=False)["flag"]
        .sum()
        .rename(columns={"flag": "remplissage_storm_event_count"})
    )
    df_out["remplissage_storm_event_count"] = df_out["remplissage_storm_event_count"] / 24
    return df_out


######### DF GLOBAL FEATURES DL ############

JOIN_KEYS = ["DEPT_ID", "harvest_year"]

VERN_FUNCTIONS_DL = [
    vernalisation_nb_jours_0_10C_dl,
    vernalisation_winter_temperature_anomaly_indicator_dl,
]

VEG_FUNCTIONS_DL = [
    vegetatif_rain_extreme_indicator_dl,
    vegetatif_humid_high_indicator_dl,
    vegetatif_rayonnement_faible_indicator_dl,
]

FLOR_FUNCTIONS_DL = [
    floraison_nb_jours_tx_gt_25_dl,
    floraison_nb_jours_tx_gt_30_dl,
    floraison_cumul_pluie_10j_max_dl,
    floraison_max_consecutive_tx_gt_25_dl,
    floraison_max_consecutive_tx_gt_30_dl,
    floraison_heat_degree_days_25_dl,
    floraison_heat_degree_days_30_dl,
    floraison_episodes_tx_gt_30_ge3_dl,
    floraison_days_heat_drought_dl,
]

REMP_FUNCTIONS_DL = [
    remplissage_nb_jours_tx_gt_25_dl,
    remplissage_heatwave_length_max_dl,
    remplissage_episodes_heatwave_ge3_dl,
    remplissage_max_consecutive_dry_days_dl,
    remplissage_storm_event_count_dl,
]

ENGINEERED_FEATURE_FUNCTIONS_DL = (
    VERN_FUNCTIONS_DL + VEG_FUNCTIONS_DL + FLOR_FUNCTIONS_DL + REMP_FUNCTIONS_DL
)

FEATURE_COMBOS_DL = {
    "vern_full": VERN_FUNCTIONS_DL,
    "veg_full":  VEG_FUNCTIONS_DL,
    "flor_full": FLOR_FUNCTIONS_DL,
    "remp_full": REMP_FUNCTIONS_DL,
    "early_stages": VERN_FUNCTIONS_DL + VEG_FUNCTIONS_DL,
    "flor_remp": FLOR_FUNCTIONS_DL + REMP_FUNCTIONS_DL,
    "all": ENGINEERED_FEATURE_FUNCTIONS_DL,
}


def build_engineered_features_mart_dl(df_meteo, combo_id="all"):
    """
    Version DL de build_engineered_features_mart.
    Agrège directement depuis meteo_hourly — pas besoin de day ni d'agrégation daily.

    Parameters
    ----------
    df_meteo : pd.DataFrame
        Sortie de add_datetime_features_dl() + add_harvest_year_dl().
        Doit contenir : datetime, saison, harvest_year, DEPT_ID
    combo_id : str
        Clé dans FEATURE_COMBOS_DL.
    """
    if combo_id not in FEATURE_COMBOS_DL:
        raise ValueError(
            f"combo_id inconnu : {combo_id!r}. "
            f"Disponibles : {list(FEATURE_COMBOS_DL.keys())}"
        )

    df_mart = None

    for feature_fn in FEATURE_COMBOS_DL[combo_id]:
        df_feat = feature_fn(df_meteo)
        feat_cols = [c for c in df_feat.columns if c not in JOIN_KEYS]
        df_feat = df_feat[JOIN_KEYS + feat_cols]

        if df_mart is None:
            df_mart = df_feat
        else:
            df_mart = df_mart.merge(df_feat, on=JOIN_KEYS, how="outer")

    return df_mart.sort_values(JOIN_KEYS).reset_index(drop=True)
