# Stereo pipeline extension to handle Planet pairs and triplets

Project in the [Remote Sensing data](https://mvaisat.wp.imt.fr) course for the MVA master

Vincent Matthys and [Jules Kozolinsky](mailto:jules.kozolinsky@ens-cachan.fr)

### Supervision
- Carlo De Franchis
- Gabriele Facciolo
- Enric Meinhardt

## Objectives
The optical S2P [1] was developed for high resolution pushbroom satellites such as Pleiades or WorldView. One of the particularities of these satellites is that the attitude of the satellite is known with high accuracy (about 50 μrad for Pleiades for instance). This implies that the pointing error in the yaw axis is locally zero, and that roll and pitch can be corrected by a translations in image space [2,3]. For satellites such as SkySat from Planet this pointing error is larger [4], which implies that the yaw error cannot be neglected anymore.
The objective of this project is to extend the S2P 3D reconstruction pipeline to handle spaceborne frame cameras such as Planet Doves and SkySat.

Dans un premier temps, estimer l'angle yaw entre entre 2 images Planet afin de pouvoir reconstruire un nuage de points 3D à partir de ces 2 images. (Espace image)
Puis, dans un second temps, avec un nombre multiples d'images, déterminer les paramètres optimaux des rotations des axes de la caméra pour la reconstruction 3D. (Espace objet)

## TP2
Dans le TP2, on se placait dans l'espace image, et on estimait les corrections avec des distances entre pixels uniquement.

## S2P
_Référence [1]_
+ Estimation des variations des angles pitch et roll : dans l'espace image, ne reposant que sur la géométrie épipolaire et les distances inter-pixels.
+ Triangulation : par paire d'image, puis fusion des nuages de points obtenus, avec l'hypothèse que chaque nuage obtenu par paire d'image est indépendant du suivant.

## Estimation de l'angle yaw
_Référence [1]_
En partant de l'estimation dans l'espace image des translations produites par la correction des angles pitch et roll (équation (6) de [1]), on espère déterminer l'angle yaw afin de reconstruire un nuage de points 3D avec une paire d'images Planet, dont le yaw ne peut plus être négligé. On considère une image référence et on estime la variation de yaw partir de celle-ci.

## Bundle adjustment
_Référence [3]_
Dans le cas général (Multi-view sterovision), on doit estimer la position, les angles caméra voire les paramètres internes de celle-ci pour chaque image. Ici, on suppose que la position (système DORIS) et les paramètres internes sont connus (calibration avant décollage dans l'espace).
Optimisation sur les coordonnées 3D des points (dans l'espace objet) pour minimiser l'erreur de reconstruction sur l'ensemble des images.

## To do

1. Compute epipolar error to find size of tiles
  Check if tile 1000x1000 still induces low error with Planet's images
2. Display figure which shows the displacements that should be applied to the matching points of the second image to make them fit on the corresponding epipolar curves.
3. Understand current implementation of pointing error

  ```python3
  # epipolar lines: 2D array of size Nx3, one epipolar line per row
  l = np.dot(x, F.T)

  # compute the error vectors (projection of xx on l)
  n = np.multiply(xx[:, 0], l[:, 0]) + np.multiply(xx[:, 1], l[:, 1]) + l[:, 2]

  d = np.square(l[:, 0]) + np.square(l[:, 1])
  a = np.divide(n, d)
  return np.vstack((np.multiply(a, l[:, 0]), np.multiply(a, l[:, 1]))).T
  ```
  Projection of xx on l ?

4. Implement correction of pointing error with the rotation
5. Extend: image space --> object space (bundle adjustment)

## Bibliographic References
[1] C. de Franchis, E. Meinhardt-Llopis, J. Michel, J-M Morel, G. Facciolo. An automatic and modular stereo pipeline for pushbroom images, ISPRS Annals, 2014

[2] H. Schmid. An analytical treatment of the problem of triangulation by stereophotogrammetry. Photogrammetria, 13:67–77, 1 1956.

[3] G. Dial and J. Grodecki. RPC replacement camera models. Proc. ASPRS Annual Conference, Baltimore, pages 1–5, 2005.

[4] [PLANET IMAGERY PRODUCT SPECIFICATIONS](https://www.planet.com/products/satellite-imagery/files/Planet_Combined_Imagery_Product_Specs_December2017.pdf)

[5] J. Sánchez. The Inverse Compositional Algorithm for Parametric Registration, IPOL, 6, 2016


## Installation for Ubuntu 16.04
```bash
conda create -n s2p python=3.6
conda install gdal
conda install ipykernel
conda install lxml
conda install requests
conda install numpy
```

Check that results for:
```bash
which pip python
```
cleary indicate the path to the s2p environment just created. Then install with pip the last 2 packages needed:

```bash
pip install utm
pip install bs4
```
You are done. Test with:
```bash
python s2p.py testdata/input_pair/config.json
```
and compile sources using:
```bash
make test
```


### Eventuellement

Ajouter dans le fichier de configuration du shell `.zshrc` :
```bash
export LD_LIBRARY_PATH=/home/XXX/anaconda3/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Et regarder si il faut mettre à jour le path python
```bash
sys.path.append('/home/XXXs/anaconda3/envs/s2p/lib/python3.6/site-packages/')
```

### PotreeConverter_PLY_toolchain

Install gcc-6 and g++-6 if necessary
```bash
git clone https://github.com/gfacciol/PotreeConverter_PLY_toolchain --recurse-submodules
CC=gcc-6 CXX=g++-6 make
```


## Usage for current developpement
Move into the conda environment
```bash
source activate s2p
```

Install needed packages
```bash
conda install ipywidgets rasterio matplotlib #needed for ipytools
```

Install a jupyter kernel for this environment
```bash
python -m ipykernel install --user --name s2p --display-name "s2p"
```

Run the pipeline
```bash
python3 s2p.py yaw_extension/config.json
```

Run Jupyter
```bash
jupyter notebook --notebook-dir=.
```

Then you can load the notebook, and select the kernel "s2p".

## How does the Satellite Stereo Pipeline work ?

> Note : With doublets only for now

### Input:
All the needed information is in a `config.json` file which contains:
+ 2 images with `TIFF`
+ 2 RPC models represented through an `XML` tree.
+ the description of the roi (region of interest)

### Pipeline

#### 1. Initialisatize and divide into tiles
<!-- ('initialisation', False), -->
(cf Section 3.4 in [1])
The epipolar error should be computed as a preliminary step, and the optimal tile size selected accordingly.


#### 2. Correct the pointing error
<!-- ('local-pointing', True),
('global-pointing', False), -->
Locally (using tiles_pairs) or globally.

#### 3. Stereo rectification
<!-- ('rectification', True), -->
Stereo rectification (or epipolar resampling) consists of resample a pair of images such that the epipolar lines become horizontal and aligned.

#### 4. Stereo Matching
<!-- ('matching', True), -->
Compute the disparity of a pair of images.


#### 5. Triangulation
(We suppose that we have only 2 images for now)
if triangulation_mode = geometric then multidisparities_to_ply
if triangulation_mode = pairwise then disparity_to_ply

#### 6. DSM Rasterization

### Output:

## Data from Planet
> Note : Only images (not video)

All images are in `.tif` format. The [Tagged Image File Format](https://fr.wikipedia.org/wiki/Tagged_Image_File_Format) (TIFF) can handle images and data within a single file.

We are provided with two different types of images:
- **Panchromatic images**: 4-band Analytic DN Image (Blue, Green, Red, NIR)
- **Pansharpened images**: 1-band Panchromatic DN Image (Pan) [Pansharpening](http://www.asprs.org/a/publications/proceedings/sandiego2010/sandiego10/Padwick.pdf) is a process of merging high-resolution panchromatic and lower resolution multispectral imagery to create a single high-resolution color image.

% Panchromatic noir et blanc

Imagery data is accompanied by Rational Polynomial Coe cients (RPCs) to enable orthorectication by the user. RPC are encoded into `.txt` files. (Projection only)

## Correction of pointing error
See [point_error_correction.pdf](https://github.com/JulesKzl/s2p/blob/master/yaw_extension/pointing_error_correction/pointing_error_correction.pdf)
