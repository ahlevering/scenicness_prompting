import json
from pathlib import Path

from osgeo import gdal
from osgeo import osr
from tqdm import tqdm
import numpy as np
from shapely.geometry import shape, GeometryCollection

import math
from time import sleep
from requests import Request
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

import rasterio
import rasterio.windows
import subprocess
from rasterio.transform import Affine

### WFS FUNCS ###
def unclip_polygon(row):
    centroid = row['geometry'].centroid
    grid_true_center_x = centroid.xy[0][0] - (centroid.xy[0][0] % 100) + 50
    grid_true_center_y = centroid.xy[1][0] - (centroid.xy[1][0] % 100) + 50
    return Point([grid_true_center_x, grid_true_center_y]).buffer(50, cap_style = 3)    

def get_scores(url, year, bbox): 
    str_bbox = ",".join(str(coord) for coord in bbox)

    layer_name = f"lbm3:clippedgridscore{year}"
    params = dict(service='WFS', version="2.0.0", request='GetFeature',
        typeName=layer_name, outputFormat='json', bbox=str_bbox, srsName="EPSG:28992", startIndex=0)
    wfs_request_url = Request('GET', url, params=params).prepare().url
    # wfs_request_url.replace("%2C", ",")
    pts_df = gpd.read_file(wfs_request_url)
    return pts_df

### WMTS FUNCS ###
def bbox_to_web_mercator(bbox):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857") 
    reproj_bbox = (*transformer.transform(bbox[1], bbox[0]), *transformer.transform(bbox[3], bbox[2]))    
    return reproj_bbox

def hotfix_name_error(wmts):
    for i, op in enumerate(wmts.operations):
        if(not hasattr(op, 'name')):
            wmts.operations[i].name = ""

def filter_row_cols_by_bbox(matrix, bbox):
    pixel_size = 0.00028 # Each pixel is assumed to be 0.28mm
    tile_size_m = matrix.scaledenominator * pixel_size

    column_orig = math.floor((float(bbox[0]) - matrix.topleftcorner[0]) / (tile_size_m * matrix.tilewidth))
    row_orig = math.floor((float(bbox[1]) - matrix.topleftcorner[1]) / (-tile_size_m * matrix.tilewidth))

    column_dest = math.floor((float(bbox[2]) - matrix.topleftcorner[0]) / (tile_size_m * matrix.tilewidth))
    row_dest = math.floor((float(bbox[3]) - matrix.topleftcorner[1]) / (-tile_size_m * matrix.tilewidth))

    if (column_orig > column_dest):
        t = column_orig
        column_orig = column_dest
        column_dest = t

    if (row_orig > row_dest):
        t = row_orig
        row_orig = row_dest
        row_dest = t

    column_dest += 1
    row_dest += 1

    return (column_orig, column_dest, row_orig, row_dest)

def calculate_output_raster_size(min_row, max_row, min_col, max_col):
    total_rows = 256 * (max_row - min_row)
    total_cols = 256 * (max_col - min_col)
    return total_rows, total_cols

def calculate_geotransform(tile_matrix, min_col, min_row):
    pixel_size = 0.00028 # Each pixel is assumed to be 0.28mm
    tile_size_m = tile_matrix.scaledenominator * pixel_size
    left = ((min_col * tile_matrix.tilewidth + 0.5) * tile_size_m) + tile_matrix.topleftcorner[0]
    top = ((min_row * tile_matrix.tileheight + 0.5) * -tile_size_m) + tile_matrix.topleftcorner[1]
    geotransform = Affine.translation(left, top) * Affine.scale(tile_size_m, -tile_size_m)
    return geotransform

def create_output_raster(out_dir, total_cols, total_rows, geotransform):
    output_raster = rasterio.open(f"{out_dir}unprojected.tiff", "w",
                                  driver="GTiff",
                                  width=total_cols,
                                  height=total_rows,
                                  count=3, # for RGB
                                  dtype=np.uint8,
                                  crs="EPSG:3857",
                                  transform=geotransform)
    return output_raster

def write_tiles_to_output_raster(wmts, wmts_layer, zoom_level, min_row, max_row, min_col, max_col, output_raster, rate_limit=True, min_row_index=0):
    min_row += min_row_index    
    for i, row in enumerate(range(min_row, max_row)):
        print(f"row {i+min_row_index} of {max_row - (min_row - min_row_index)}")
        for col in range(min_col, max_col):
            try: # Wrapping to skip random corrupted tiles
                tile = wmts.gettile(
                    layer=wmts_layer,
                    tilematrixset="GoogleMapsCompatible",
                    tilematrix=zoom_level,
                    row=row,
                    column=col,
                    resampling='bilinear',
                    format='image/jpeg'
                )

                # Read the tile data and store it in the output raster
                img = rasterio.io.MemoryFile(tile).open().read()
                output_raster.write(img, window=rasterio.windows.Window(
                    col * 256 - min_col * 256,
                    row * 256 - min_row * 256,
                    256, 256))
            except Exception as e:
                print(e)
            if rate_limit > 0:
                sleep(0.05)

def reproject_raster(out_dir, filename_prefix, out_pixel_size):
    reproj_cmd = f"gdalwarp -t_srs EPSG:28992 -s_srs EPSG:3857 -tr {out_pixel_size} {out_pixel_size} {out_dir}unprojected.tiff {out_dir}{filename_prefix}_raster.tiff"
    subprocess.call(reproj_cmd, shell=True)                
    subprocess.call(f"rm {out_dir}unprojected.tiff", shell=True) # Remove unprojected raster

def load_geojson_polys(polys_file):
    with open(polys_file) as f:
        features = json.load(f)["features"]
    geoms = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in features])    
    records = [f['properties'] for f in features]
    return geoms, records

def lookup_grid_scores(polys_file, grid_ids):
    _, records = load_geojson_polys(polys_file)
    grid_scores = {}
    
    for entry in records:
        if entry['gridcode'] in grid_ids:
            grid_scores[str(entry['gridcode'])] = entry
    return grid_scores