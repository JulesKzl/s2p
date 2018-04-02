# Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@cmla.ens-cachan.fr>
# Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2015, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>
# Copyright (C) 2015, Julien Michel <julien.michel@cnes.fr>


from __future__ import print_function
import os
import numpy as np

from s2plib import sift
from s2plib import rectification
from s2plib import rpc_utils
from s2plib import rpc_model
from s2plib import common
from s2plib import estimation
from s2plib import evaluation
from s2plib.config import cfg

import pdb

def evaluation_from_estimated_F(im1, im2, rpc1, rpc2, x, y, w, h, A=None,
        matches=None):
    """
    Measures the pointing error on a Pleiades' pair of images, affine approx.

    Args:
        im1, im2: paths to the two Pleiades images (usually jp2 or tif)
        rpc1, rpc2: two instances of the rpc_model.RPCModel class
        x, y, w, h: four integers defining the rectangular ROI in the first image.
            (x, y) is the top-left corner, and (w, h) are the dimensions of the
            rectangle.
        A (optional): 3x3 numpy array containing the pointing error correction
            for im2.
        matches (optional): Nx4 numpy array containing a list of matches to use
            to compute the pointing error

    Returns:
        the mean pointing error, in the direction orthogonal to the epipolar
        lines. This error is measured in pixels, and computed from an
        approximated fundamental matrix.
    """
    if not matches:
        matches = sift.matches_on_rpc_roi(im1, im2, rpc1, rpc2, x, y, w, h)
    p1 = matches[:, 0:2]
    p2 = matches[:, 2:4]
    print('%d sift matches' % len(matches))

    # apply pointing correction matrix, if available
    if A is not None:
        p2 = common.points_apply_homography(A, p2)

    # estimate the fundamental matrix between the two views
    rpc_matches = rpc_utils.matches_from_rpc(rpc1, rpc2, x, y, w, h, 5)
    F = estimation.affine_fundamental_matrix(rpc_matches)

    # compute the mean displacement from epipolar lines
    d_sum = 0
    for i in range(len(p1)):
        x  = np.array([p1[i, 0], p1[i, 1], 1])
        xx = np.array([p2[i, 0], p2[i, 1], 1])
        ll  = F.dot(x)
        #d = np.sign(xx.dot(ll)) * evaluation.distance_point_to_line(xx, ll)
        d = evaluation.distance_point_to_line(xx, ll)
        d_sum += d
    return d_sum/len(p1)


def rigid_transform_matrix(v, ab):
    """
    Build the matrix of an rigid transform in homogeneous coordinates.

    Arguments:
        v: numpy 1D array, of length 3, containing the parameters of the
            planar transform.

    Returns:
        A numpy 3x3 matrix, representing the euclidean transform (rotation
        followed by translation) in homogeneous coordinates.
    """
    # Unpack v
    # Translation independent
    if (len(ab) == 0):
        tx = v[0]
        ty = v[1]
        if (len(v) > 2):
            theta = v[2]
        else:
            theta = 0
    # Translation dependent
    else:
        t = v[0]
        tx = t*ab[0]
        ty = t*ab[1]
        if (len(v) > 1):
            theta = v[1]
        else:
            theta = 0

    # # centering of coordinates with respect to the center of the big pleiades
    # # image (40000x40000)
    C = np.eye(3)
    if (len(v) > 3):
        C[0, 2] = v[3]
        C[1, 2] = v[4]

    # matrix construction
    R = np.eye(3)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R[0, 0] =  cos_theta
    R[0, 1] = -sin_theta
    R[1, 0] =  sin_theta
    R[1, 1] =  cos_theta

    T = np.eye(3)
    T[0, 2] = tx
    T[1, 2] = ty
    # return np.linalg.inv(C).dot(R).dot(T).dot(C)
    # return np.dot(C, np.dot(R, T))
    return np.dot(R, T)


def cost_function(v, *args):
    """
    Objective function to minimize in order to correct the pointing error.

    Arguments:
        v: vector containing the parameters of the rigid
            transformation we are looking for.
        rpc1, rpc2: two instances of the rpc_model.RPCModel class
        matches: 2D numpy array containing a list of matches. Each line
            contains one pair of points, ordered as x1 y1 x2 y2.
            The coordinate system is the one of the big images.
        alpha: relative weight of the error terms: e + alpha*(h-h0)^2. See
            paper for more explanations.

    Returns:
        The sum of pointing errors and altitude differences, as written in the
        paper formula (1).
    """
    F, matches = args[0], args[1]
    apply_rectification = args[2]

    v_copy = v.copy()

    # verify that parameters are in the bounding box
    # if (np.abs(v[2]) > 200*np.pi or
    #     np.abs(v[0]) > 10000 or
    #     np.abs(v[1]) > 10000):
    #     print('warning: cost_function is going too far')
    #     print(v)

    # compute the altitudes from the matches without correction
    x1 = matches[:, 0]
    y1 = matches[:, 1]
    x2 = matches[:, 2]
    y2 = matches[:, 3]

    # compute epipolar lines
    p1 = np.column_stack((matches[:,0:2],np.ones((len(matches),1))))
    Fx1 = np.transpose(np.dot(F, np.transpose(p1)))


    # Compute normal vector to epipolar liness
    if (apply_rectification):
        ab = np.array([Fx1[0][0], Fx1[0][1]])
        ab = ab/np.linalg.norm(ab)
        if len(v) > 2:
            v_copy[1] /= 10

    else:
        ab = []
        if len(v) > 3:
            v_copy[2] /= 10

    # transform the coordinates of points in the second image according to
    # matrix A, built from vector v
    A = rigid_transform_matrix(v_copy, ab)
    p2 = np.column_stack((matches[:,2:4],np.ones((len(matches),1))))

    Ax2 = np.transpose(np.dot(A, np.transpose(p2)))

    # Compute distance between a point and a line (vectorize)
    d = np.divide(np.abs(np.sum(np.multiply(Ax2, Fx1), 1)), np.linalg.norm(Fx1[:,0:2], axis=1))

    # compute the cost
    cost = np.sum(d)
    return cost

def print_params(v):
    """
    Display the pointing correction parameters.

    Args:
        v: 1D numpy array of length 3

    This function is called by the fmin_bfgs optimization function at each
    iteration, to display the current values of the parameters.
    """
    print("v:", v)

def local_transformation(r1, r2, x, y, w, h, m,
                         pointing_error_correction_method='analytic', apply_rectification=True):
    """
    Estimates the optimal rotation following a translation to minimise the
    relative pointing error on a given tile.

    Args:
        r1, r2: two instances of the rpc_model.RPCModel class
        x, y, w, h: region of interest in the reference image (r1)
        m: Nx4 numpy array containing a list of matches, one per line. Each
            match is given by (p1, p2, q1, q2) where (p1, p2) is a point of the
            reference view and (q1, q2) is the corresponding point in the
            secondary view.

    Returns:
        3x3 numpy array containing the homogeneous representation of the
        optimal planar rotation composed with translation, to be applied
        to the secondary image in order to correct the pointing error.
    """
    # estimate the affine fundamental matrix between the two views
    n = cfg['n_gcp_per_axis']
    rpc_matches = rpc_utils.matches_from_rpc(r1, r2, x, y, w, h, n)
    F = estimation.affine_fundamental_matrix(rpc_matches)

    args_opti = (F, m, apply_rectification)

    # Solve the resulting optimization problem
    from scipy.optimize import fmin_l_bfgs_b

    number_variables = pointing_error_correction_method
    if (not apply_rectification):
        number_variables += 1

    v0 = np.zeros(number_variables)
    v, min_val, debug = fmin_l_bfgs_b(
            cost_function,
            v0,
            args=args_opti,
            approx_grad=True,
            factr=1,
            #bounds=[(-150, 150), (-100, 100), (-100, 100)])
            #maxiter=50,
            callback=print_params)
            #iprint=0,
            #disp=0)


    # compute epipolar lines
    p1 = np.column_stack((m[:,0:2],np.ones((len(m),1))))
    Fp1 = np.dot(F, np.transpose(p1)).T


    if (apply_rectification):
        # Compute normal vector to epipolar liness
        ab = np.array([Fp1[0][0], Fp1[0][1]])
        ab = ab/np.linalg.norm(ab)

        # Compute translation
        t = v[0]
        tx = t*ab[0]
        ty = t*ab[1]
        # theta
        if (pointing_error_correction_method > 1):
            theta = v[1]
            theta /= 10

        else:
            theta = 0
    else:
        ab = []
        tx = v[0]
        ty = v[1]
        # theta
        if (pointing_error_correction_method > 1):
            theta = v[2]
            theta /= 10
        else:
            theta = 0

    print('tx: %f' % tx)
    print('ty: %f' % ty)
    print('theta: %f' % theta)
    if (len(v) > 3):
        print('cx: %f' % v[3])
        print('cy: %f' % v[4])
    print('min cost: %f' % min_val)
    print(debug)
    A = rigid_transform_matrix(v, ab)
    print('A:', A)
    return A, F

def local_translation_rotation(r1, r2, x, y, w, h, m):
    """
    Estimates the optimal translation to minimise the relative pointing error
    on a given tile.

    Args:
        r1, r2: two instances of the rpc_model.RPCModel class
        x, y, w, h: region of interest in the reference image (r1)
        m: Nx4 numpy array containing a list of matches, one per line. Each
            match is given by (p1, p2, q1, q2) where (p1, p2) is a point of the
            reference view and (q1, q2) is the corresponding point in the
            secondary view.

    Returns:
        3x3 numpy array containing the homogeneous representation of the
        optimal planar translation, to be applied to the secondary image in
        order to correct the pointing error.
    """
    # estimate the affine fundamental matrix between the two views
    n = cfg['n_gcp_per_axis']
    rpc_matches = rpc_utils.matches_from_rpc(r1, r2, x, y, w, h, n)
    F = estimation.affine_fundamental_matrix(rpc_matches)

    # Compute rectification homographies
    # S1, S2 = estimation.rectifying_similarities_from_affine_fundamental_matrix(F, cfg['debug'])
    hmargin = cfg['horizontal_margin']
    vmargin = cfg['vertical_margin']
    S1, S2, F_rect = rectification.rectification_homographies(rpc_matches, x, y, w, h, hmargin, vmargin)

    N = len(m)
    # Apply rectification on image 1
    S1p1 = common.points_apply_homography(S1, m[:,0:2])
    S1p1 = np.column_stack((S1p1,np.ones((N,1))))
    print("Points 1 rectied")
    print(S1p1[:10])
    # Apply rectification on image 2
    S2p2 = common.points_apply_homography(S1, m[:,2:4])
    S2p2 = np.column_stack((S2p2, np.ones((N,1))))
    print("Points 2 rectied")
    print(S2p2[:10])

    # Compute F in the rectified space
    rect_matches = np.column_stack((S1p1[:,0:2], S2p2[:, 0:2]))
    F_rect2 = estimation.affine_fundamental_matrix(rect_matches)

    # Compute epipolar lines
    FS1p1 = np.dot(F_rect2, S1p1.T).T
    print(FS1p1[:10])

    # Normalize epipolar lines
    c1 = -FS1p1[:, 1]
    FS1p1_norm = FS1p1/c1[:, np.newaxis]
    print(FS1p1_norm[:10])


    # Variable of optimization problem
    A_ = np.ones((N,1))
    # A_ = np.column_stack((S2p2[:,0].reshape(N, 1),np.ones((N,1))))
    # b_ = S2p2[:, 1] - FS1p1_norm[:, 2]
    b_ = S2p2[:, 1] - S1p1[:, 1]
    t_med = np.median(b_)
    print(t_med)

    t_med2 = np.sort(b_)[int(N/2)]
    print("t_med2", t_med2)

    # min ||Ax + b||^2 => x = - (A^T A )^-1 A^T b
    # X_ = - np.dot(np.linalg.inv(np.dot(A_.T, A_)), np.dot(A_.T, b_))
    # [theta, t] = X_
    # print(t, theta)
    # t = X_[0]

    # Compute epipolar lines witout recitifcation
    p1 = np.column_stack((m[:,0:2],np.ones((N,1))))
    Fp1 = np.dot(F, p1.T).T

    # Compute normal vector to epipolar liness
    ab = np.array([Fp1[0][0], Fp1[0][1]])
    ab = ab/np.linalg.norm(ab)

    tx = t_med*ab[0]
    ty = t_med*ab[1]
    print(tx, ty)

    # Get translation in not rectified image
    T = common.matrix_translation(0, -t_med)
    T = np.linalg.inv(S2).dot(T).dot(S2)
    # T = np.dot(S2_inv, T)
    print(T)

    theta = 0
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    A = np.array([[cos_theta, -sin_theta, tx],
                  [sin_theta,  cos_theta, ty],
                  [0, 0, 1]])
    return A, F

def local_translation_rectified(r1, r2, x, y, w, h, m):
    """
    Estimates the optimal translation to minimise the relative pointing error
    on a given tile.

    Args:
        r1, r2: two instances of the rpc_model.RPCModel class
        x, y, w, h: region of interest in the reference image (r1)
        m: Nx4 numpy array containing a list of matches, one per line. Each
            match is given by (p1, p2, q1, q2) where (p1, p2) is a point of the
            reference view and (q1, q2) is the corresponding point in the
            secondary view.

    Returns:
        3x3 numpy array containing the homogeneous representation of the
        optimal planar translation, to be applied to the secondary image in
        order to correct the pointing error.
    """
    # estimate the affine fundamental matrix between the two views
    n = cfg['n_gcp_per_axis']
    rpc_matches = rpc_utils.matches_from_rpc(r1, r2, x, y, w, h, n)
    F = estimation.affine_fundamental_matrix(rpc_matches)

    # Apply rectification on image 1
    S1p1 = common.points_apply_homography(S1, m[:,0:2])
    S1p1 = np.column_stack((S1p1,np.ones((N,1))))
    print("Points 1 rectied")
    print(S1p1[:10])
    # Apply rectification on image 2
    S2p2 = common.points_apply_homography(S1, m[:,2:4])
    S2p2 = np.column_stack((S2p2, np.ones((N,1))))
    print("Points 2 rectied")
    print(S2p2[:10])

    # Compute F in the rectified space
    rect_matches = np.column_stack((S1p1[:,0:2], S2p2[:, 0:2]))
    F_rect2 = estimation.affine_fundamental_matrix(rect_matches)

    # Compute epipolar lines
    FS1p1 = np.dot(F_rect2, S1p1.T).T
    print(FS1p1[:10])

    # Normalize epipolar lines
    c1 = -FS1p1[:, 1]
    FS1p1_norm = FS1p1/c1[:, np.newaxis]
    print(FS1p1_norm[:10])


    b_ = np.abs(S2p2[:, 1] - S1p1[:, 1])
    t_med = np.median(b_)
    print(t_med)

    t_med2 = np.sort(b_)[int(N/2)]
    print("t_med2", t_med2)

    # Compute epipolar lines witout recitifcation
    p1 = np.column_stack((m[:,0:2],np.ones((N,1))))
    Fp1 = np.dot(F, p1.T).T

    # Compute normal vector to epipolar liness
    ab = np.array([Fp1[0][0], Fp1[0][1]])
    ab = ab/np.linalg.norm(ab)

    tx = t_med*ab[0]
    ty = t_med*ab[1]
    print(tx, ty)

    # Get translation in not rectified image
    T = common.matrix_translation(0, -t_med)
    T = np.linalg.inv(S2).dot(T).dot(S2)
    # T = np.dot(S2_inv, T)
    print(T)

    # the correction to be applied to the second view is the opposite
    A = np.array([[1, 0, -out_x],
                  [0, 1, -out_y],
                  [0, 0, 1]])
    return A, F



def error_vectors(m, F, ind='ref'):
    """
    Computes the error vectors for a list of matches and a given fundamental
    matrix.

    Args:
        m: Nx4 numpy array containing a list of matches, one per line. Each
           match is given by (x, y, x', y') where (x, y) is a point of the
           reference view and (x', y') is the corresponding point in the
           secondary view.
        F: fundamental matrix between the two views
        ind (optional, default is 'ref'): index of the image on which the lines
            are plotted to compute the error vectors. Must be either 'ref' or
            'sec' (reference or secondary image)

    Returns:
        Nx2 numpy array containing a list of planar vectors. Each vector is
        obtained as the difference between x' and its projection on the
        epipolar line Fx.
    """
    # divide keypoints in two lists: x (first image) and xx (second image), and
    # convert them to homogeneous coordinates
    N  = len(m)
    x  = np.ones((N, 3))
    xx = np.ones((N, 3))
    x [:, 0:2] = m[:, 0:2]
    xx[:, 0:2] = m[:, 2:4]

    # epipolar lines: 2D array of size Nx3, one epipolar line per row
    if ind == 'sec':
        l = np.dot(x, F.T)
    elif ind == 'ref':
        l = np.dot(xx, F)
    else:
        print("pointing_accuracy.error_vectors: invalid 'ind' argument")

    # compute the error vectors (going from the projection of x or xx on l to x
    # or xx)
    if ind == 'sec':
        n = np.multiply(xx[:, 0], l[:, 0]) + np.multiply(xx[:, 1], l[:, 1]) + l[:, 2]
    else:
        n = np.multiply(x[:, 0], l[:, 0]) + np.multiply(x[:, 1], l[:, 1]) + l[:, 2]
    d = np.square(l[:, 0]) + np.square(l[:, 1])
    a = np.divide(n, d)
    return np.vstack((np.multiply(a, l[:, 0]), np.multiply(a, l[:, 1]))).T


def local_translation(r1, r2, x, y, w, h, m):
    """
    Estimates the optimal translation to minimise the relative pointing error
    on a given tile.

    Args:
        r1, r2: two instances of the rpc_model.RPCModel class
        x, y, w, h: region of interest in the reference image (r1)
        m: Nx4 numpy array containing a list of matches, one per line. Each
            match is given by (p1, p2, q1, q2) where (p1, p2) is a point of the
            reference view and (q1, q2) is the corresponding point in the
            secondary view.

    Returns:
        3x3 numpy array containing the homogeneous representation of the
        optimal planar translation, to be applied to the secondary image in
        order to correct the pointing error.
    """
    # estimate the affine fundamental matrix between the two views
    n = cfg['n_gcp_per_axis']
    rpc_matches = rpc_utils.matches_from_rpc(r1, r2, x, y, w, h, n)
    F = estimation.affine_fundamental_matrix(rpc_matches)

    # compute the error vectors
    e = error_vectors(m, F, 'sec')

    # compute the median: as the vectors are collinear (because F is affine)
    # computing the median of each component independently is correct
    N = len(e)
    out_x = np.sort(e[:, 0])[int(N/2)]
    out_y = np.sort(e[:, 1])[int(N/2)]

    # the correction to be applied to the second view is the opposite
    A = np.array([[1, 0, -out_x],
                  [0, 1, -out_y],
                  [0, 0, 1]])
    return A, F


def compute_correction(img1, rpc1, img2, rpc2, x, y, w, h, pec_method='analytic',
                       pec_degree=1, apply_rectification=True):
    """
    Computes pointing correction matrix for specific ROI

    Args:
        img1: path to the reference image.
        rpc1: paths to the xml file containing the rpc coefficients of the
            reference image
        img2: path to the secondary image.
        rpc2: paths to the xml file containing the rpc coefficients of the
            secondary image
        x, y, w, h: four integers defining the rectangular ROI in the reference
            image. (x, y) is the top-left corner, and (w, h) are the dimensions
            of the rectangle. The ROI may be as big as you want. If bigger than
            1 Mpix, only five crops will be used to compute sift matches.

    Returns:
        a 3x3 matrix representing the planar transformation to apply to img2 in
        order to correct the pointing error, and the list of sift matches used
        to compute this correction.
    """
    # read rpcs
    r1 = rpc_model.RPCModel(rpc1)
    r2 = rpc_model.RPCModel(rpc2)

    m = sift.matches_on_rpc_roi(img1, img2, r1, r2, x, y, w, h)

    print(pec_method)
    if m is not None:
        if (pec_method == 'analytic'):
            if (pec_degree == 1):
                if (apply_rectification):
                    A, F = local_translation_rectified(r1, r2, x, y, w, h, m)
                else:
                    A, F = local_translation(r1, r2, x, y, w, h, m)
            elif (pec_degree == 2):
                A, F = local_translation_rotation(r1, r2, x, y, w, h, m)
            else:
                raise ValueError("Analytic method for degree > 2 is not implemented")
        elif (pec_method == 'numeric'):
            A, F = local_transformation(r1, r2, x, y, w, h, m, pec_degree, apply_rectification)
        else:
            raise ValueError("Choose between analytic or numeric solver for correcting the pointing error")


    else:
        A = None
        F = None

    return A, m, F


def from_next_tiles(tiles, ntx, nty, col, row):
    """
    Computes the pointing correction of a specific tile using the pointing
    corrections of its neighbors. We assume that all pointing corrections are
    translations.

    Args:
        tiles: list of paths to folders associated to each tile
        ntx: number of tiles in horizontal direction
        nty: number of tiles in vertical direction
        col: horizontal index (from 1 to ntx) of the current tile
        row: horizontal index (from 1 to nty) of the current tile

    Returns:
        the estimated pointing correction for the specified tile
    """
    # TODO
    return None


def global_from_local(tiles):
    """
    Computes the pointing correction of a full roi using local corrections on
    tiles.

    Args:
        tiles: list of paths to folders associated to each tile

    Returns:
        the estimated pointing correction for the specified tile

    In each folder we expect to find the files pointing.txt and center.txt. The
    file pointing.txt contains the local correction (a projective transform
    given in homogeneous coordinates), and the file center.txt contains the
    coordinates of the mean of the keypoints of the secondary image.
    """
    # lists of matching points
    x  = []
    xx = []

    # loop over all the tiles
    for f in tiles:
        center = os.path.join(f, 'center_keypts_sec.txt')
        pointing = os.path.join(f, 'pointing.txt')
        if os.path.isfile(center) and os.path.isfile(pointing):
            A = np.loadtxt(pointing)
            p = np.loadtxt(center)
            if A.shape == (3, 3) and p.shape == (2,):
                q = np.dot(A, np.array([p[0], p[1], 1]))
                x.append(p)
                xx.append(q[0:2])

    if not x:
        return np.eye(3)
    elif len(x) == 1:
        return A
    elif len(x) == 2:
        #TODO: replace translation with similarity
        return estimation.translation(np.array(x), np.array(xx))
    else:
        # estimate an affine transformation transforming x in xx
        return estimation.affine_transformation(np.array(x), np.array(xx))
