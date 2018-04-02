cloud=$1
# cloud=testdata/cloud.ply
aply=out.ply
blas=out.las

# make a text ply
if head $cloud | grep "format binary" ; then
   echo "converting binary ply to ascii"
   ../PotreeConverter_PLY_toolchain/plytool/ply2ascii <$cloud> $aply
   cloud=$aply
fi

# conert ply to las because PotreeConverter is not able to read PLY
# (ATTENTION to the -parse field s means skip)
#./PotreeConverter/LAStools/bin/txt2las -parse xyzsssRGBs -verbose  -i  $cloud -o $blas 2>/dev/null
../PotreeConverter_PLY_toolchain/PotreeConverter/LAStools/bin/txt2las -parse xyzRGB -verbose -i  $cloud -o $blas

# # convert to potree
# # copy the template
# cp -r PotreeConverter/PotreeConverter/resources PotreeConverter/build/PotreeConverter/
# mv PotreeConverter/build/PotreeConverter/resources/page_template/viewer_template{_flip,}.html

# run  the conversion
export LC_ALL=C
../PotreeConverter_PLY_toolchain/PotreeConverter/build/PotreeConverter/PotreeConverter $blas -o . -p testPotree --edl-enabled --material ELEVATION --overwrite

# # clean
# rm $aply $blas
#
# echo ""
# echo "Launch a simple webserver and visit http://localhost:8000/testdata"
# echo "> python -m SimpleHTTPServer &"
#
