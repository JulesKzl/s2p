1. yaw n'est pas vraiment le problème ?
2. vrai problème de sift sur les planets:
  d2_0009: sift_matches_pointing et sift_matches_disp. Le disp contient vraiment des sift pas bons
3. Bundle adjustment next step : variables d'intérêts ? Reprendre l'équation à implémenter ?
4. Remonter à une region terrestre à partir des coefficients RPC ?

> Localilsation avec une altitude constante (classe rpc.localisation) (done)

> ligne 365 rectification: matrice A avec pour origine (0,0) de l'image complète

> Risque d'utilisation de l'optimiseur avec trop de degrés de liberté : perte du sens physique

> Variante de SIFT: ASIFT bcp plus robuste: https://github.com/opencv/opencv/blob/master/samples/python/asift.py

> Verification avec bfgs: restreindre l'optimization à une seule translation et comparer avec la médiane. Est-ce le fait de laisser une rotation possible qui perturbe ?

> Normaliser les entrées de bfgs pour que la rotation soit comparable  la translation

> Option dans config.json pour autoriser la rotation ou pas

> Tester sur testdata pour s'assurer que l'optimiseur ne trouve pas de rotation (quasi nulle) sur testdata et que la normalisation fonctionne bien. (effet de la translation très proche de celle obtenue par la médiane)

> Bundle adjustment : minimisation de l'erreur de reprojection $E=\sum_{i,j}d(x_{i,j}, DK(R_j T_j)X_i)^2$. On cherche des n-uplets de correspondance entre les images pour minimiser cette erreur de reprojection, sachant que $R_j$ et $T_j$ sont les paramètres à mimiser en parallèle de X_i. Hartley Zisserman : ouvrage de référence (chapitre 17 ou 18 suivant la version)
