# Contexte

# Interlocuteur officiel
Carlo De Franchis <carlodef@gmail.com>

# Objectifs
Dans un premier temps, estimer l'angle yaw entre entre 2 images Planet afin de pouvoir reconstruire un nuage de points 3D à partir de ces 2 images. Puis, dans un second temps, avec un nombre multiples d'images, déterminer les paramètres optimaux des rotations des axes de la caméra pour la reconstruction 3D

# TP2
Dans le TP2, on se placait dans l'espace image, et on estimait les corrections avec des distances entre pixels uniquement.

# S2P
_Référence [1]_
+ Estimation des variations des angles pitch et roll : dans l'espace image, ne reposant que sur la géométrie épipolaire et les distances inter-pixels.
+ Triangulation : par paire d'image, puis fusion des nuages de points obtenus, avec l'hypothèse que chaque nuage obtenu par paire d'image est indépendant du suivant.

# Estimation de l'angle yaw
_Référence [1]_
En partant de l'estimation dans l'espace image des translations produites par la correction des angles pitch et roll (équation (6) de [1]), on espère déterminer l'angle yaw afin de reconstruire un nuage de points 3D avec une paire d'images Planet, dont le yaw ne peut plus être négligé. On considère une une image référence et on estime la variation de yaw partir de celle-ci.

# Bumdle adjustment
_Référence [3]_
Dans le cas général (cd multi-view sterovision, Monasse), on doit estimer la position, les angles caméra voire les paramètres internes de celle-ci pour chaque image. Ici, on suppose que la position (système DORIS) et les paramètres internes sont connus (calibration avant décollage dans l'espace).
Optimisation sur les coordonnées 3D des points (dans l'espace objet) pour minimiser l'erreur de reconstruction sur l'ensemble des images.

# To do
1. Développer l'équation (6) de [1] avec la rotation, optimisation et  implémentation.
2. L'incorporer à S2P et réussir à reconstruire un nuage de points 3D pour une paire d'image Planet
3. Extension au bundle adjustment

# Bibliographie
[1] C. de Franchis, E. Meinhardt-Llopis, J. Michel, J-M Morel, G. Facciolo. An automatic and modular stereo pipeline for pushbroom images, ISPRS Annals, 2014

[2] H. Schmid. An analytical treatment of the problem of triangulation by stereophotogrammetry. Photogrammetria, 13:67–77, 1 1956.

[3] G. Dial and J. Grodecki. RPC replacement camera models. Proc. ASPRS Annual Conference, Baltimore, pages 1–5, 2005.

[4] [PLANET IMAGERY PRODUCT SPECIFICATIONS](https://www.planet.com/products/satellite-imagery/files/Planet_Combined_Imagery_Product_Specs_December2017.pdf)

[5] J. Sánchez. The Inverse Compositional Algorithm for Parametric Registration, IPOL, 6, 2016
