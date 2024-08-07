import math
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.stats import ttest_ind


## T-TEST ENSEMBLING LAND COVER ORDERING ##
ensemble_file = "data/outputs/ensembling/ensembling_preds_clip.gpkg"
ensemble_results = gpd.read_file(ensemble_file)

corine_file = "data/son_pts_aux_info.geojson"
corine_pts = gpd.read_file(corine_file)

corine_pts = corine_pts[["ID", "Average", "lc"]]
corine_pts["lc"] = corine_pts["lc"].str[:2]

corine_pts = corine_pts.merge(ensemble_results, on="ID")

comparisons = [['12', '13'], ['13', '14'], ['21', '22'], ['32', '33'], ['51', '52']]

for comparison in comparisons:
    series1 = list(corine_pts[corine_pts['lc'].isin([comparison[0]])]['preds_2'].values)
    series2 = list(corine_pts[corine_pts['lc'].isin([comparison[1]])]['preds_2'].values)
    t_stat, p_value = ttest_ind(series1, series2, equal_var=False)
    print(f"Comparison {comparison}: t-statistic = {t_stat}, p-value = {p_value}")

## T-TEST METRICS 25 SAMPLE CASE ##
results_path_25 = "data/outputs/contrastive/clip/zero_shot_contrasts_25.csv"
results_path_full = "data/outputs/contrastive/clip/zero_shot_contrasts.csv"

df_25 = pd.read_csv(results_path_25)

comparisons = [['A photo of an area that is extremely', 'A photo of an area that is']]
for comparison in comparisons:
    series1 = list(df_25[df_25['context'].isin([comparison[0]])]['kendalls_tau'].values)
    series2 = list(df_25[df_25['context'].isin([comparison[1]])]['kendalls_tau'].values)
    t_stat, p_value = ttest_ind(series1, series2, equal_var=False)
    print(f"Comparison {comparison}: t-statistic = {t_stat}, p-value = {p_value}")