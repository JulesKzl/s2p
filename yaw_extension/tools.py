import numpy as np
from glob import glob
from tqdm import tqdm
import json
from ipytools import readGTIFF
from ipytools import writeGTIFF
from ipytools import readGTIFFmeta
from ipytools import gdal_resample_image_to_longlat
from ipytools import gdal_get_longlat_of_pixel
from ipytools import clickablemap

# Display tool
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ipytools import display_RSO

################################################################################
############################## GLOBALS #########################################
RELATIVE_PATH = "../data/"
# Video
VIDEO = RELATIVE_PATH + "s02_20150507T020554Z/"
# n_uplet
n_uplet = RELATIVE_PATH + "s03_20161003T16*/"
n_uplet_panchromatic = n_uplet + "panchromatic/"
n_uplet_pansharp = n_uplet + "pansharp/*"
################################################################################

def s03_handler(offset = 2):
    """
    Handler for S03 Planet data

    Input:
    ------
    offset: integer
            level of verbosity for satellite names
    """
    lst_tif = glob(n_uplet_panchromatic + "*.tif")
    areas = {}
    n_uplets = {}
    for img in lst_tif:
        splitted_ = img.split("_")
        # Current area name
        cur_area = splitted_[-2]
        # Current area part
        cur_part  = int(splitted_[-1].split(".")[0])
        # Current satellite
        cur_satellite = splitted_[-4][-offset:]
        # Build areas dictionary
        if cur_area not in areas.keys():
            areas[cur_area] = {}
        if cur_part not in areas[cur_area].keys():
            areas[cur_area][cur_part] = {}
        if cur_satellite not in areas[cur_area][cur_part].keys():
            areas[cur_area][cur_part][cur_satellite] = img
        # Build n_uplets dictionary
        if cur_satellite not in n_uplets.keys():
            n_uplets[cur_satellite] = {}
        if cur_area not in n_uplets[cur_satellite].keys():
            n_uplets[cur_satellite][cur_area] = {}
        if cur_part not in n_uplets[cur_satellite][cur_area].keys():
            n_uplets[cur_satellite][cur_area][cur_part] = img
    return lst_tif, areas, n_uplets

def data_handler(area,  lst_tif = s03_handler()[0], roi = None, verbosity = True):
    """
    Returns the n-uplet corresponding to area with the corresponding labels

    Input:
    ------
    area:    str
             pattern to search into lst_tif
    lst_tif: list of str
             list of tif files
    roi:     dictionary
               keys: [x, y, w, h]
               values: integers
             region of interest

    Output:
    -------
    out:
    labels:
    """
    out = []
    labels = []
    if roi is not None:
        assert all(key in roi.keys() for key in ["x", "y", "w", "h"]),\
        "Missing keys for roi"
        x, w, y, h = roi['x'], roi['w'], roi['y'], roi['h']
    for img in tqdm(lst_tif):
        if area in img:
            labels += [img]
            if roi is not None:
                out += [display_RSO(readGTIFF(img)[y:y+h, x:x+w], plot = False)]
            else:
                out += [display_RSO(readGTIFF(img), plot = False)]
    if verbosity:
        for i in range(len(labels)):
            print (labels[i])
    return out, labels

def plot_mosaic(n_uplet, area, n_views = 5, plot = True, save = False, outfile = None):
    """
    Plot a mosaic of n-uplets
    """
    L = {key : list(n_uplet[key][area].values()) for key in n_uplet.keys()}
    for key in L.keys():
        L[key].sort()
    satellites = list(L.keys())
    satellites.sort()
    # satellites = ['1Z']
    satellites = ['1Z', '8Z', '7Z']
    nbr_satellite = len(satellites)

    f = plt.figure(figsize = (8 * nbr_satellite, 3 * n_views))
    gs = gridspec.GridSpec(n_views, nbr_satellite)
    # set the spacing between axes.
    gs.update(wspace=0.01, hspace=0.02)

    for j in tqdm(range(len(satellites))):
        lst = L[satellites[j]]
        for i in range(n_views):
            ax = plt.subplot(gs[i, j])
            plt.axis('off')
    #         ax.set_xticklabels([])
    #         ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.imshow(display_RSO(readGTIFF(lst[i]), plot = False).squeeze(), cmap = "gray")

    if save == True:
        try:
            f.savefig(outfile,
                      bbox_inches='tight',
                      pad_inches=0)
        except OError as e:
            print ("I/O error({0}): {1}".format(e.errno, e.strerror))
    if plot == True:
        plt.show()

def get_image_longlat_polygon(fname):
    """
    Determines the image size and computes the long/lat coordinates
    of the four image corners.

    Input:
    ------
    fname: str
           GeoTIFF filename
    Output:
    -------
    poly:  dictionary (keys: 'type' and 'Polygon')
           a GeoJSON polygon with the four corners
    """
    # get the image size
    A = readGTIFFmeta(fname)
    H = int(A[0]['height'])
    W = int(A[0]['width'])

    cols, rows = np.meshgrid([0,W], [0,H])
    cols = [0,W,W,0]
    rows = [0,0,H,H]

    coords = gdal_get_longlat_of_pixel(fname, cols, rows, verbose=False)

    # remove the altitude (which is 0)
    coords = [ [x[0],x[1]] for x in coords ]

    poly = {'type': 'Polygon', 'coordinates': [coords]}
    return poly


def reprojection(img, tmp_img = None, tmp_png = None,
                 display = False, verbose = False):
    """
    TO DO: Reprojection of list of IMG
    DIFFICULTIES: Vbox returned by overlaymap, not map instance
    """
    if tmp_img is None:
        tmp_img = "tmp/" + img.split("pan_")[-1]
    if tmp_png is None:
        tmp_png = tmp_img.replace(".tif", ".png")

    ## Reproject the image in longlat (one GeoTIFF and one PNG)
    gdal_resample_image_to_longlat(img, tmp_img)

    # Extract the new image footprint from the GeoTIFF
    footprint = get_image_longlat_polygon(tmp_img)

    if verbose:
        print(footprint)

    if display:
        # Convert TIFF to 8-bits and write a PNG
        data = readGTIFF(tmp_img)
        writeGTIFF(display_RSO(data, plot = False), tmp_png)
        # Display the reprojected PNG overlaid on a map at the coordinates of the footprint
        mo = ipytools.overlaymap(footprint0, tmp_png , zoom = 13)
        display(mo)
    return footprint

def geolocalisation(trace):
    """
    """
    if type(trace) == dict:
        imgs = list(trace.values())
        imgs.sort()
        imgs
    else:
        assert type(trace) == list, "Trace should be a list or a dict of images"

    footprint = []
    for i in tqdm(imgs):
        footprint.append(get_image_longlat_polygon(i))

    m = clickablemap()

    for i in range(len(imgs)):
        m.add_GeoJSON(footprint[i])

    m.center = footprint[0]['coordinates'][0][0][::-1]
    display(m)

    return footprint

def workaround_json_int64(o):
    if isinstance(o,np.integer) : return int(o)
    raise TypeError

def write_json(duet, roi=None):
    """
    duet = triplets['d1']['0001'][:2]
    """
    # Get current config file
    with open('config.json', 'r') as f:
        user_cfg = json.load(f)
    user_cfg['out_dir'] = './output'
    # Config path of images and RPC
    user_cfg['images'] = [
        {
            "img": duet[i],
            "rpc": duet[i].replace(".tif", "_rpc.txt")
        }
        for i in range(len(duet))]
    # Config ROI
    if (roi == None):
        user_cfg["full_img"] = True
    else:
        user_cfg["full_img"] = False
        user_cfg["roi"] = roi
    # Modify config file
    with open('config.json', 'w') as f:
        json.dump(user_cfg, f, indent=2, default=workaround_json_int64)
