"""
Optical satellites geometric modeling tools

Copyright (C) 2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
Copyright (C) 2018, Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>
"""

from __future__ import print_function
import numpy as np
import requests
import subprocess
import datetime
import sys
import bs4
import os
import cv2

import ipytools


def is_absolute(url):
    return bool(requests.utils.urlparse(url).netloc)


def listFD(url, ext=''):
    page = requests.get(url).text
    soup = bs4.BeautifulSoup(page, 'html.parser')
    files = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

    parsed_url = requests.utils.urlparse(url)
    base = '%s://%s' % (parsed_url.scheme, parsed_url.netloc)
    return [x if is_absolute(x) else base+x for x in files]


def acquisition_date(geotiff_path):
    """
    """
    with ipytools.rio_open(geotiff_path) as src:
        date_string = src.tags()['NITF_STDIDC_ACQUISITION_DATE']
        return datetime.datetime.strptime(date_string, "%Y%m%d%H%M%S")


def rpc_from_geotiff(geotiff_path, outrpcfile='.rpc'):
    """
    """
    env = os.environ.copy()
    if geotiff_path.startswith(('http://', 'https://')):
        env['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = geotiff_path[-3:]
        path = '/vsicurl/{}'.format(geotiff_path)
    else:
        path = geotiff_path

    f = open(outrpcfile, 'wb')
    x = subprocess.Popen(["gdalinfo", path], stdout=subprocess.PIPE).communicate()[0]
    x = x.splitlines()
    for l in x:

        if(1):
            if (b'SAMP_' not in l) and (b'LINE_' not in l) and (b'HEIGHT_' not in l) and (b'LAT_' not in l) and (b'LONG_' not in l) and (b'MAX_' not in l) and (b'MIN_' not in l):
                  continue
            y = l.strip().replace(b'=',b': ')
            if b'COEFF' in y:
                  z = y.split(b' ')
                  t=1
                  for j in z[1:]:
                          f.write(b'%s_%d: %s\n'%(z[0][:-1],t,j))
                          t+=1
            else:
                  f.write((y+b'\n'))
    f.close()
    return rpc_model.RPCModel(outrpcfile)


def get_angle_from_cos_and_sin(c, s):
    """
    Computes x in ]-pi, pi] such that cos(x) = c and sin(x) = s.
    """
    if s >= 0:
        return np.arccos(c)
    else:
        return -np.arccos(c)


def matrix_translation(x, y):
    t = np.eye(3)
    t[0, 2] = x
    t[1, 2] = y
    return t


def points_apply_homography(H, pts):
    """
    Applies an homography to a list of 2D points.

    Args:
        H: numpy array containing the 3x3 homography matrix
        pts: numpy array containing the list of 2D points, one per line

    Returns:
        a numpy array containing the list of transformed points, one per line
    """
    # if the list of points is not a numpy array, convert it
    if (type(pts) == list):
        pts = np.array(pts)

    # convert the input points to homogeneous coordinates
    if len(pts[0]) < 2:
        print("""points_apply_homography: ERROR the input must be a numpy array
          of 2D points, one point per line""")
        return
    pts = np.hstack((pts[:, 0:2], pts[:, 0:1]*0+1))

    # apply the transformation
    Hpts = (np.dot(H, pts.T)).T

    # normalize the homogeneous result and trim the extra dimension
    Hpts = Hpts * (1.0 / np.tile( Hpts[:, 2], (3, 1)) ).T
    return Hpts[:, 0:2]


def bounding_box2D(pts):
    """
    bounding box for the points pts
    """
    dim = len(pts[0])  # should be 2
    bb_min = [min([t[i] for t in pts]) for i in range(dim)]
    bb_max = [max([t[i] for t in pts]) for i in range(dim)]
    return bb_min[0], bb_min[1], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1]


def image_crop_gdal(inpath, x, y, w, h, outpath):
    """
    Image crop defined in pixel coordinates using gdal_translate.

    Args:
        inpath: path to an image file
        x, y, w, h: four integers defining the rectangular crop pixel coordinates.
            (x, y) is the top-left corner, and (w, h) are the dimensions of the
            rectangle.
        outpath: path to the output crop
    """
    if int(x) != x or int(y) != y:
        print('WARNING: image_crop_gdal will round the coordinates of your crop')

    env = os.environ.copy()
    if inpath.startswith(('http://', 'https://')):
        env['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = inpath[-3:]
        path = '/vsicurl/{}'.format(inpath)
    else:
        path = inpath

    cmd = ['gdal_translate', path, outpath,
           '-srcwin', str(x), str(y), str(w), str(h),
           '-ot', 'Float32',
           '-co', 'TILED=YES',
           '-co', 'BIGTIFF=IF_NEEDED']

    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        if inpath.startswith(('http://', 'https://')):
            if not requests.head(inpath).ok:
                print('{} is not available'.format(inpath))
                return
        print('ERROR: this command failed')
        print(' '.join(cmd))
        print(e.output)


def match_pair(fa, fb):
    a = ipytools.readGTIFF(fa)
    a = TP1_solutions.simple_equalization_8bit(a)
    b = ipytools.readGTIFF(fb)
    b = TP1_solutions.simple_equalization_8bit(b)

    # KP
    sift = cv2.xfeatures2d.SIFT_create()
    #kp = sift.detect(a,None)
    kp1, des1 = sift.detectAndCompute(a,None)
    #kp = sift.detect(a,None)
    kp2, des2 = sift.detectAndCompute(b,None)
#    img=cv2.drawKeypoints(a,kp,b)
#    display_image(img)
    #cv2.imwrite('sift_keypoints.jpg',img)


    #https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)


    # cv2.drawMatchesKnn expects list of lists as matches.
    #img3 = cv2.drawMatchesKnn(a,kp1,b,kp2,good,a,flags=2)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    #display_image(img3)
    return  pts1, pts2
