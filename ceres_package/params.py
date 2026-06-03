import os
import numpy as np

##################  VARIABLES  ##################
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")



##################  CONSTANTS  #####################
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")



#################  Departements  ####################
DEPARTEMENTS = {
            '01': 'Ain',
            '02': 'Aisne',
            '03': 'Allier',
            '04': 'Alpes-de-Haute-Provence',
            '05': 'Hautes-Alpes',
            '06': 'Alpes-Maritimes',
            '07': 'Ardèche',
            '08': 'Ardennes',
            '09': 'Ariège',
            '10': 'Aube',
            '11': 'Aude',
            '12': 'Aveyron',
            '13': 'Bouches-du-Rhône',
            '14': 'Calvados',
            '15': 'Cantal',
            '16': 'Charente',
            '17': 'Charente-Maritime',
            '18': 'Cher',
            '19': 'Corrèze',
            '21': "Côte-d'Or",
            '22': "Côtes-d'Armor",
            '23': 'Creuse',
            '24': 'Dordogne',
            '25': 'Doubs',
            '26': 'Drôme',
            '27': 'Eure',
            '28': 'Eure-et-Loir',
            '29': 'Finistère',
            '2A': 'Corse-du-Sud',
            '2B': 'Haute-Corse',
            '30': 'Gard',
            '31': 'Haute-Garonne',
            '32': 'Gers',
            '33': 'Gironde',
            '34': 'Hérault',
            '35': 'Ille-et-Vilaine',
            '36': 'Indre',
            '37': 'Indre-et-Loire',
            '38': 'Isère',
            '39': 'Jura',
            '40': 'Landes',
            '41': 'Loir-et-Cher',
            '42': 'Loire',
            '43': 'Haute-Loire',
            '44': 'Loire-Atlantique',
            '45': 'Loiret',
            '46': 'Lot',
            '47': 'Lot-et-Garonne',
            '48': 'Lozère',
            '49': 'Maine-et-Loire',
            '50': 'Manche',
            '51': 'Marne',
            '52': 'Haute-Marne',
            '53': 'Mayenne',
            '54': 'Meurthe-et-Moselle',
            '55': 'Meuse',
            '56': 'Morbihan',
            '57': 'Moselle',
            '58': 'Nièvre',
            '59': 'Nord',
            '60': 'Oise',
            '61': 'Orne',
            '62': 'Pas-de-Calais',
            '63': 'Puy-de-Dôme',
            '64': 'Pyrénées-Atlantiques',
            '65': 'Hautes-Pyrénées',
            '66': 'Pyrénées-Orientales',
            '67': 'Bas-Rhin',
            '68': 'Haut-Rhin',
            '69': 'Rhône',
            '70': 'Haute-Saône',
            '71': 'Saône-et-Loire',
            '72': 'Sarthe',
            '73': 'Savoie',
            '74': 'Haute-Savoie',
            '76': 'Seine-Maritime',
            '77': 'Seine-et-Marne',
            '78': 'Yvelines',
            '79': 'Deux-Sèvres',
            '80': 'Somme',
            '81': 'Tarn',
            '82': 'Tarn-et-Garonne',
            '83': 'Var',
            '84': 'Vaucluse',
            '85': 'Vendée',
            '86': 'Vienne',
            '87': 'Haute-Vienne',
            '88': 'Vosges',
            '89': 'Yonne',
            '90': 'Territoire de Belfort',
            '91': 'Essonne',
            '92': 'Hauts-de-Seine',
            '93': 'Seine-Saint-Denis',
            '94': 'Val-de-Marne',
            '95': "Val-d'Oise",
            }

##############  DATA METEO FRANCE  ##################
COLONNES_BLE = [
        # Identifiants & localisation
        "NUM_POSTE","LAT", "LON", "ALTI", "AAAAMMJJHH",
        # Température
        "T", "TN", "TX", "TD", "TNSOL", "TCHAUSSEE", "T10", "T20", "T50", "T100",
        "DG",
        # Humidité
        "U", "UN", "HUN", "UX", "HUX", "DHUMI80", "DHUMI40", "DHUMEC", "TSV",
        # Précipitations
        "RR1", "DRR1",
        # Vent
        "FF", "FXI", "DD",
        # Rayonnement & ensoleillement
        "GLO", "INS",
        # Neige & sol
        "NEIGETOT", "SOL",
        # Pression
        "PMER"
        ]

RENAME_COLONNES_BLE = {
        # Identifiants & localisation
        "NUM_POSTE":  "id_station",
        "LAT":        "latitude",
        "LON":        "longitude",
        "ALTI":       "altitude_m",
        "DATE": "datetime",
        # Température
        "T":          "temp_air_c",
        "TN":         "temp_min_c",
        "TX":         "temp_max_c",
        "TD":         "temp_rosee_c",
        "TNSOL":      "temp_min_sol_10cm_c",
        "TCHAUSSEE":  "temp_surface_sol_c",
        "T10":        "temp_sol_10cm_c",
        "T20":        "temp_sol_20cm_c",
        "T50":        "temp_sol_50cm_c",
        "T100":       "temp_sol_100cm_c",
        "DG":         "duree_gel_min",
        # Humidité
        "U":          "humidite_relative_pct",
        "UN":         "humidite_min_pct",
        "HUN":        "heure_humidite_min",
        "UX":         "humidite_max_pct",
        "HUX":        "heure_humidite_max",
        "DHUMI80":    "duree_humidite_sup80_min",
        "DHUMI40":    "duree_humidite_inf40_min",
        "DHUMEC":     "duree_humectation_foliaire_min",
        "TSV":        "tension_vapeur_hpa",
        # Précipitations
        "RR1":        "precipitations_1h_mm",
        "DRR1":       "duree_precipitations_min",
        # Vent
        "FF":         "vent_moyen_10m_ms",
        "FXI":        "rafale_max_ms",
        "DD":         "direction_vent_deg",
        # Rayonnement
        "GLO":        "rayonnement_global_jcm2",
        "INS":        "insolation_min",
        # Neige & sol
        "NEIGETOT":   "hauteur_neige_sol_cm",
        "SOL":        "etat_sol",
        # Pression
        "PMER":       "pression_mer_hpa",
    }

###############  DATA SOIL GRID  ####################


#################  DATA NVDI  #######################
