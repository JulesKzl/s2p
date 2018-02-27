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
1. Développer l'équation (6) de [1] avec la rotation, optimisation et  implémentation.
2. L'incorporer à S2P et réussir à reconstruire un nuage de points 3D pour une paire d'image Planet
    1. Understand the data we get from Planet (cf `understand_s2p.py`)
        - Panchromatic or pansharp images, different cameras
    1. Try to understand the way tiles are divided from the roi
    2. Launch current s2p with 2 images of Planet = adapt `config.json` file
        - size of tile ?
        - utm box ?

    3. Implement new equation
3. Extend to bundle adjustment

## Bibliographic References
[1] C. de Franchis, E. Meinhardt-Llopis, J. Michel, J-M Morel, G. Facciolo. An automatic and modular stereo pipeline for pushbroom images, ISPRS Annals, 2014

[2] H. Schmid. An analytical treatment of the problem of triangulation by stereophotogrammetry. Photogrammetria, 13:67–77, 1 1956.

[3] G. Dial and J. Grodecki. RPC replacement camera models. Proc. ASPRS Annual Conference, Baltimore, pages 1–5, 2005.

[4] [PLANET IMAGERY PRODUCT SPECIFICATIONS](https://www.planet.com/products/satellite-imagery/files/Planet_Combined_Imagery_Product_Specs_December2017.pdf)

[5] J. Sánchez. The Inverse Compositional Algorithm for Parametric Registration, IPOL, 6, 2016


## Installation sur ubuntu 16.04
```bash
conda create -n s2p python=3.6
conda install gdal
conda install ipykernel
conda install lxml
conda install requests
conda install numpy
```

Vérifier que les résultats de :
```bash
which pip python
```
tombent bien dans l'environnement s2p créé, puis :

```bash
pip install utm
pip install bs4
```
et enfin tester :
```bash
python s2p.py testdata/input_pair/config.json
```
ou
```bash
make test
```


### Eventuellement

Ajouter dans le fichier de configuration du shell `.zshrc` :
```
export LD_LIBRARY_PATH=/home/vmatthys/anaconda3/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Et regarder si il faut mettre à jour le path python
```
sys.path.append('/home/vmatthys/anaconda3/envs/s2p/lib/python3.6/site-packages/)
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

Imagery data is accompanied by Rational Polynomial Coe cients (RPCs) to enable orthorectication by the user. RPC are encoded into `.txt` files. (Only inverse model ?)
