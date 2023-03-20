from pathlib import Path
from h3 import h3
import pandas as pd
import geopandas as gpd
import numpy as np
import csv
import pickle
from sklearn.cluster import KMeans

from codebase.pt_funcs.dataloaders import PP2DataContainer
from codebase.utils.file_utils import load_csv

np.random.seed(113)

def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        matches = pickle.load(f)
    return matches

# Filepaths
labels_file = "data/photo_df_scores.csv"
images_root = "../../../data/datasets/PlacePulse2/final_photo_dataset/"
ratings = "../../../data/datasets/PlacePulse2/metadata/votes.csv"
embeddings_file = "data/embeddings/ViT-L-14_pp2.pkl"
bins_file = "data/PP2_pts_with_bins.geojson"

data = PP2DataContainer(labels_file).labels
all_votes = load_csv(ratings)
all_votes = pd.DataFrame(all_votes)
all_votes.columns = all_votes.iloc[0]
all_votes = all_votes[1:]
embeddings = load_pickle(embeddings_file)

to_remove = []
img_root = "../../../data/datasets/PlacePulse2/final_photo_dataset/"
for pt in all_votes['left_id']:
    if not Path(img_root+f"{pt}.jpg").is_file():
        to_remove.append(pt)
for pt in all_votes['right_id']:
    if not Path(img_root+f"{pt}.jpg").is_file():
        to_remove.append(pt)        

all_votes = all_votes[~all_votes['left_id'].isin(to_remove)]
all_votes = all_votes[~all_votes['right_id'].isin(to_remove)]
all_votes.to_csv("data/votes.csv")

if not Path(bins_file).exists():
    km = KMeans(n_clusters=25)
    all_embeddings = np.stack(list(embeddings.values()))
    kmeans = km.fit_predict(list(all_embeddings))    
    # splits_gdf = splits_gdf[:1000]
    resolution = 4

    def assign_hexbin(row):
        row['bin'] = h3.geo_to_h3(row['Lat'], row['Lon'], resolution)
        return row

    pts_with_bins = data.apply(assign_hexbin, axis=1) 
    pts_with_bins['centroid_k'] = kmeans 

    pts_with_bins.to_file(bins_file, driver="GeoJSON")
else:
    pts_with_bins = gpd.read_file(bins_file)

out_dir = Path(f"data/splits_pp2/")
out_dir.mkdir(exist_ok=True, parents=True)

all_bins = set(set(pts_with_bins['bin']))

votes_with_metadata = all_votes.merge(pts_with_bins, how="inner", left_on="left_id", right_on="Picture")
# votes_with_metadata = votes_with_metadata[votes_with_metadata['category'].isin(['safety'])]
for num in [25, 50, 75, 100, 175, 250, 325, 400, 500]:
    # Sample each group
    grouped_votes = pd.DataFrame(votes_with_metadata.groupby(["bin", "category"]))
    sampled_groups = grouped_votes[1].sample(n=num, random_state=113, replace=False)
    sampled_comparisons = []
    for k in sampled_groups:
        random_index = np.random.randint(0, len(k)) # Pick a random index in the group
        sampled_comparisons.append(k.iloc[random_index])
    sampled_comparisons_df = pd.DataFrame(sampled_comparisons)
    merged_ids = sampled_comparisons_df['left_id'] + '_' + sampled_comparisons_df['right_id']

    with open(f"{out_dir}/{num}.csv", 'w', newline='') as out_file:
        wr = csv.writer(out_file, quoting=csv.QUOTE_ALL)
        wr.writerow(merged_ids.values)