import os
import json

import ipytools

def get_files_triplets(folder):
    """  Get all TIFF in the folder """
    tiff_dict = {1: [], 2:[], 3:[]}
    for dirname, dirnames, filenames in os.walk(folder):
        # print path to all filenames.
        for filename in filenames:
            if  filename.endswith('.tif'):
                complete_filename = os.path.join(dirname, filename)
                d = int(filename.split('_')[3][1])
                tiff_dict[d].append(complete_filename)
    for key in tiff_dict:
        tiff_dict[key] = sorted(tiff_dict[key])
    return tiff_dict

def retrieve_triplets():
    """ Build triplets from data """
    folders = {'1107':{}, '1148':{}, '1231':{}}
    for f in folders:
        folder = '../data/s03_20161003T16' + f + 'Z/panchromatic/'
        folders[f] = get_files_triplets(folder)
    return folders

def choose_triplets(folders, d, i, indices):
    im_list = []
    for f in indices:
        im_name = folders[f][d][i-1]
        im_list.append(im_name)
    return im_list


def retrieve_video():
    """  Get all TIFF in the folder """
    im_list = []
    folder = '../data/s02_20150507T020554Z/video_frames/'
    for dirname, dirnames, filenames in os.walk(folder):
        # print path to all filenames.
        for filename in filenames:
            if  filename.endswith('.tif'):
                complete_filename = os.path.join(dirname, filename)
                im_list.append(complete_filename)
    print(str(len(im_list)) + ' images')
    return sorted(im_list)

def choose_video(folders, indices):
    im_list = []
    for f in indices:
        im_name = folders[f]
        im_list.append(im_name)
    return im_list



def display_images(images_name, roi=None):
    """ Display chosen triplet """
    l = []
    g = []
    for im_name in images_name:
        g.append(im_name)
        im = ipytools.readGTIFF(im_name)
        if (roi != None):
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            im = im[y:y+h, x:x+w]
        print('Dimension: ', im.shape)
        l.append(ipytools.simple_equalization_8bit(im))
    ipytools.display_gallery(l, g)

def workaround_json_int64(o):
    if isinstance(o,np.integer) : return int(o)
    raise TypeError

def write_json(images_name, num_variable=0, tile_size=500, roi=None, rpc_name=[]):
    """ Write JSON with chosen triplets """
    # Get current config file
    with open('config.json', 'r') as f:
        user_cfg = json.load(f)
    user_cfg['out_dir'] = './output'
    # Config path of images and RPC
    images = []
    for i, img_name in enumerate(images_name):
        img_dict = {}
        img_dict['img'] = img_name
        if (len(rpc_name) > 0):
            img_dict['rpc'] = rpc_name[i]
        else:
            img_dict['rpc'] = '..' + img_name.split('.')[2] + '_rpc.txt'
        images.append(img_dict)
    user_cfg['images'] = images
    # Tile size
    user_cfg["tile_size"] = tile_size
    # Config ROI
    if (roi == None):
        user_cfg["full_img"] = True
    else:
        user_cfg["full_img"] = False
        user_cfg["roi"] = roi
    # Number of variable to optimize when correction pointing error
    # 0 = use translation
    # 2 = only translation enable in optimizer
    # 3 = Translation + Rotation
    # 5 = Translation + Rotation + center
    # Modify config file
    user_cfg["n_optim_variables"] = num_variable
    with open('config.json', 'w') as f:
        json.dump(user_cfg, f, indent=2, default=workaround_json_int64)
