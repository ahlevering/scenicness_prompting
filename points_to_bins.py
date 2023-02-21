from pathlib import Path
from h3 import h3
import pandas as pd
import geopandas as gpd
import numpy as np
import csv

np.random.seed(113)

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

out_dir = Path(f"data/splits/")
out_dir.mkdir(exist_ok=True, parents=True)

all_bins = set(set(pts_with_bins['bin']))

for num in [25, 50, 75, 100, 175, 250, 325, 400, 500]:
    # Sample data from set
    averages = pts_with_bins['Average'].to_list()
    # sampled_values = np.random.normal(low=np.mean(averages), scale=np.std(averages), size=num)
    sampled_values = np.random.uniform(low=np.min(averages), high=np.max(averages), size=num)
    values_clipped = np.clip(sampled_values, 1, 10) # Ensure values are within dataset bounds

    bins = []
    sampled_pts = []
    for sample in sampled_values:

        bin_included = False
        i = 0
        closest_pts = pts_with_bins.iloc[(pts_with_bins['Average']-(sample)).abs().argsort()]
        # Look for closest point within 1% of the dataset
        while not bin_included and i < 0.01 * len(pts_with_bins):
            closest_pt = closest_pts.iloc[i]
            # bins_not_in_split = pts_with_bins['bin'].isin(bins).any()
            if not closest_pt['bin'] in bins:
                sampled_pts.append(int(closest_pt['ID']))
                bins.append(closest_pt['bin'])
                bin_included = True
            else:
                i += 1
        if not bin_included:
            closest_pt = closest_pts.iloc[0]
            sampled_pts.append(int(closest_pt['ID']))

    # sampled_pts = pd.DataFrame(columns=closest_pt.keys() , data=sampled_pts)
    # sampled_pts = gpd.GeoDataFrame( sampled_pts,
    #                                 geometry=gpd.points_from_xy(sampled_pts['Lat'], sampled_pts['Lon']),
    #                                 crs=4326)
    with open(f"{out_dir}/{num}.csv", 'w', newline='') as out_file:
        wr = csv.writer(out_file, quoting=csv.QUOTE_ALL)
        wr.writerow(sampled_pts)