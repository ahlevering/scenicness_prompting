from pathlib import Path
from h3 import h3
import geopandas as gpd
import numpy as np
import csv
import pickle
from sklearn.cluster import KMeans

np.random.seed(113)
def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        matches = pickle.load(f)
    return matches

embeddings_file = "data/embeddings.pkl"
embeddings = load_pickle(embeddings_file)

km = KMeans(n_clusters=25)
all_embeddings = np.stack(list(embeddings.values()))
kmeans = km.fit_predict(list(all_embeddings))

bins_file = "data/son_pts_with_bins.geojson"
if not Path(bins_file).exists():
    splits_file = "../son_det/data/source/son_splits.geojson"
    splits_gdf = gpd.read_file(splits_file)
    # splits_gdf = splits_gdf[:1000]
    resolution = 4

    def assign_hexbin(row):
        row['bin'] = h3.geo_to_h3(row['Lat'], row['Lon'], resolution)
        return row

    pts_with_bins = splits_gdf.apply(assign_hexbin, axis=1) 
    pts_with_bins.to_file(bins_file, driver="GeoJSON")
else:
    pts_with_bins = gpd.read_file(bins_file)

pts_with_bins['centroid_k'] = kmeans


out_dir = Path(f"data/splits/")
out_dir.mkdir(exist_ok=True, parents=True)

all_bins = set(set(pts_with_bins['bin']))

# # Get baseline sample ids
# train_samples = pts_with_bins.sample(21210) # 10% for training

# with open(f"{out_dir}/baseline_train.csv", 'w', newline='') as out_file:
#     wr = csv.writer(out_file, quoting=csv.QUOTE_ALL)
#     wr.writerow(list(train_samples['ID'].values))

# remainder = pts_with_bins[~pts_with_bins['ID'].isin(train_samples['ID'])]
# val_samples = pts_with_bins.sample(10605) # 5% for validation    

# with open(f"{out_dir}/baseline_val.csv", 'w', newline='') as out_file:
#     wr = csv.writer(out_file, quoting=csv.QUOTE_ALL)
#     wr.writerow(list(val_samples['ID'].values))

for num in [25, 50, 75, 100, 175, 250, 325, 400, 500]:
    # Sample data from set
    n_samples = round(num/25)
    sampled_pts = pts_with_bins.groupby(["centroid_k"]).sample(n=n_samples, random_state=113, replace=False)

    with open(f"{out_dir}/{num}.csv", 'w', newline='') as out_file:
        wr = csv.writer(out_file, quoting=csv.QUOTE_ALL)
        wr.writerow(sampled_pts['ID'].values)