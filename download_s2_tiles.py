import numpy as np
from owslib.wmts import WebMapTileService
from pathlib import Path
from codebase.pt_funcs.dataloaders import SoNDataContainer
from tqdm import tqdm

from codebase.utils.geodata_utils import * # unclip_polygon, get_scores, filter_row_cols_by_bbox

year = "18"
out_pixel_size = 10
filename_prefix = 's2'
out_dir = f"data/patches/s2_tiles/"
Path(out_dir).mkdir(exist_ok=True, parents=True)

labels_file = "data/son_pts_with_bins.geojson"
labels_gdf = SoNDataContainer(labels_file).labels

wmts_layer = f"s2cloudless-2018_3857"
zoom_level = 13

# Set-up WMTS service
wmts = WebMapTileService("https://tiles.maps.eox.at/wmts/")
hotfix_name_error(wmts)
tile_matrix = wmts.tilematrixsets["GoogleMapsCompatible"].tilematrix[str(zoom_level)]

# Get extent of entire S2 tile
dataset_extent = labels_gdf.total_bounds
dataset_bbox = bbox_to_web_mercator(dataset_extent)
# bbox = [dataset_bbox[0]-1120, dataset_bbox[1]+1120, dataset_bbox[2]+1120, dataset_bbox[3]-1120]
min_col, max_col, min_row, max_row = filter_row_cols_by_bbox(tile_matrix, dataset_bbox)

total_rows = 256 * (max_row - min_row)
total_cols = 256 * (max_col - min_col)

tile = wmts.gettile(
    layer=wmts_layer,
    tilematrixset="GoogleMapsCompatible",
    tilematrix=zoom_level,
    row=min_row,
    column=min_col,
    format='image/jpeg'
)

output_raster_path = out_dir+"unprojected.tiff"
if not Path(output_raster_path).exists():
    # Calculate transformation parameters
    geotransform = calculate_geotransform(tile_matrix, min_col, min_row)        
    output_raster = create_output_raster(out_dir, total_cols, total_rows, geotransform)
else:
    output_raster = rasterio.open(output_raster_path, mode='r+')

# loop through the tiles and write them to the output raster
write_tiles_to_output_raster(wmts, wmts_layer, zoom_level, min_row, max_row, min_col, max_col, output_raster, min_row_index=167)
output_raster.close()

    # Reproject raster to Dutch national system
    # Didn't manage to quickly do this with rasterio.
    # Using command-line GDAL instead, which means a temporary 2x storage space requirement.
    # Sorry.
    # reproj_cmd = f"gdalwarp -t_srs EPSG:28992 -s_srs EPSG:3857 -tr {out_pixel_size} {out_pixel_size} {out_dir}unprojected.tiff {out_dir}{filename_prefix}_raster.tiff"
    # subprocess.call(reproj_cmd, shell=True)
    # subprocess.call("rm tiles/unprojected.tiff", shell=True)