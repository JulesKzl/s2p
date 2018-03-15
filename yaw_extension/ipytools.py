"""
* simplified map interaction using ipyleaflet
* display images and tables in the notebook
* wrappers for gdaltransform and gdalwarp

Copyright (C) 2017-2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

from __future__ import print_function
import ipywidgets

### simplified map interaction using ipyleaflet

def clickablemap(center = [48.790153, 2.327395], zoom = 13,
                 layout = ipywidgets.Layout(width='100%', height='500px') ):
    # look at: http://leaflet.github.io/Leaflet.draw/docs/examples/basic.html

    import os
    import json

    from ipyleaflet import (
        Map,
        Rectangle,
        Polygon,
        TileLayer, ImageOverlay,
        DrawControl, GeoJSON
    )

    #%matplotlib inline
 #   %matplotlib notebook


    # google tileserver 
    # https://stackoverflow.com/questions/9394190/leaflet-map-api-with-google-satellite-layer 
    mosaicsTilesURL = 'https://mt1.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}' # Hybrid: s,h; Satellite: s; Streets: m; Terrain: p;

    # Map Settings 
    # Define colors
    colors = {'blue': "#009da5"}
    # Define initial map center lat/long
    #center = [48.790153, 2.327395]
    # Define initial map zoom level
    #zoom = 13
    # Create the map
    m = Map(
        center = center, 
        zoom = zoom,
        scroll_wheel_zoom = True,
        default_tiles = TileLayer(url=mosaicsTilesURL), # using custom basemap 
        layout = layout
    )

    # Define the draw tool type options
    polygon = {'shapeOptions': {'color': colors['blue']}}
    rectangle = {'shapeOptions': {'color': colors['blue']}} 

    ## Create the draw controls
    ## @see https://github.com/ellisonbg/ipyleaflet/blob/master/ipyleaflet/leaflet.py#L293
    #dc = DrawControl(
    #    polygon = polygon,
    #    rectangle = rectangle
    #)
    dc = DrawControl(polygon={'shapeOptions': {'color': '#0000FF'}}, 
                     polyline={'shapeOptions': {'color': '#0000FF'}},
                     circle={'shapeOptions': {'color': '#0000FF'}},
                     rectangle={'shapeOptions': {'color': '#0000FF'}},
                     )
    
    
    # Initialize an action counter variable
    m.actionCount = 0
    m.AOIs = []

    
    # Register the draw controls handler
    def handle_draw(self, action, geo_json):
        # Increment the action counter
        #global actionCount
        m.actionCount += 1
        # Remove the `style` property from the GeoJSON
        geo_json['properties'] = {}
        # Convert geo_json output to a string and prettify (indent & replace ' with ")
        geojsonStr = json.dumps(geo_json, indent=2).replace("'", '"')
        m.AOIs.append (json.loads(geojsonStr))


    # Attach the draw handler to the draw controls `on_draw` event
    dc.on_draw(handle_draw)
    m.add_control(dc)
    
    # add a custom function to create and add a Rectangle layer 
    # (LESS USEFUL THAN add_geojson)
    def add_rect(*args, **kwargs):
        r = Rectangle( *args, **kwargs)
        return m.add_layer(r)
    m.add_rectangle = add_rect 
    
    # add a custom function to create and add a Polygon layer 
    def add_geojson(*args, **kwargs):
        # ugly workaround to call without data=aoi
        if 'data' not in kwargs:
            kwargs['data'] = args[0]
            args2=[i for i in args[1:-1]]
        else:
            args2=args

        r = GeoJSON( *args2, **kwargs)
        return m.add_layer(r)
    m.add_GeoJSON = add_geojson 
    
    # Display
    return m


def overlaymap(aoiY, imagesurls, zoom = 13,
               layout = ipywidgets.Layout(width='100%', height='500px') ):
    
    import os
    import json
    import numpy as np
    
    from ipyleaflet import (
        Map,
        Rectangle,
        Polygon,
        TileLayer, ImageOverlay,
        DrawControl, 
    )

    ## handle the case of imageurls not a list
    if type(imagesurls) != list:
        imagesurls = [imagesurls]
        
    number_of_images = len(imagesurls)
    
    ## handle both kinds of calls with aoi, or aoi['coordinates']
    if 'coordinates' in aoiY:
        aoiY=aoiY['coordinates'][0]
        
        
    # create the Map object    
    # google tileserver 
    # https://stackoverflow.com/questions/9394190/leaflet-map-api-with-google-satellite-layer 
    mosaicsTilesURL = 'https://mt1.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}' # Hybrid: s,h; Satellite: s; Streets: m; Terrain: p;
    m = Map( center = aoiY[0][::-1] , 
            zoom = zoom,
            scroll_wheel_zoom = True,
            default_tiles = TileLayer(url=mosaicsTilesURL, opacity=1.00),  # use custom tiles
            layout = layout,
       )

    #vlayer = VideoOverlay(videoUrl, videoBounds )
    #m.add_layer(vlayer)



    ### this shows an animated gif
    #m.add_layer(layer)


    # display map (this show)
    #display(m)



    ############## ADD INTERACTIVE LAYER
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets

    
    # meke sure that the images have unique names 
    imagesurls =  ['%s?%05d'%(i,np.random.randint(10000)) for i in imagesurls] 

    
    # draw bounding polygon
    y = [ a[::-1] for a in aoiY ]
    p = Polygon(locations=y, weight=2, fill_opacity=0.25)
    m.add_layer(p)

    # create image 
    layer = ImageOverlay(url='%s'%(imagesurls[0]), bounds=[ list(np.max(aoiY,axis=0)[::-1]) , list(np.min(aoiY,axis=0)[::-1]) ])

    m.add_layer(layer)

    # callback fro flipping images
    def showim(i):
            if(i<len(imagesurls)):
   #      -----       FLICKERS ----
   #             layer.url='%s'%(imagesurls[i])
   #             layer.visible = False
   #             layer.visible = True
    
   #    ALTERNATIVE:  add a new layer 
                layer = ImageOverlay(url='%s'%(imagesurls[i]), bounds=[ list(np.max(aoiY,axis=0)[::-1]) , list(np.min(aoiY,axis=0)[::-1]) ])
                m.add_layer(layer)
                # remove old ones 
                if len(m.layers)>30: # image buffer 
                    for l in (m.layers[1:-1]):
                        m.remove_layer(l)
    
    
    # build the UI
    #interact(showim,i=len(imagesurls)-1)
        #interact(showim, i=widgets.IntSlider(min=0,max=len(imagesurls),step=1,value=0));
    play = widgets.Play(
        interval=200,   #ms
        value=0,
        min=0,
        max=len(imagesurls)-1,
        step=1,
        description="Press play",
        disabled=False,
    )
    slider = widgets.IntSlider( min=0, max=len(imagesurls)-1, description='Frame:')
    label  = widgets.Label(value="")
    def on_value_change(change):
        label.value=imagesurls[change['new']] 
        showim(change['new'])
    slider.observe(on_value_change, 'value')
    b1 = widgets.Button(description='fw', layout=widgets.Layout(width='auto') )
    b2 = widgets.Button(description='bw', layout=widgets.Layout(width='auto'))
    b3 = widgets.Button(description='hide', layout=widgets.Layout(width='auto'))
    b4 = widgets.Button(description='hidePoly', layout=widgets.Layout(width='auto'))
    def clickfw(b):
            slider.value=slider.value+1
    def clickbw(b):
        slider.value=slider.value-1
    def clickhide(b):
        if layer.visible:
            layer.visible = False
        else:
            layer.visible = True
    def clickhidePoly(b):
        if p.visible:
            p.visible = False
        else:
            p.visible = True
    b1.on_click( clickfw )
    b2.on_click( clickbw )
    b3.on_click( clickhide )
    b4.on_click( clickhidePoly )

    
    # add a custom function to create and add a Polygon layer 
    def add_geojson(*args, **kwargs):
        # ugly workaround to call without data=aoi
        if 'data' not in kwargs:
            kwargs['data'] = args[0]
            args2=[i for i in args[1:-1]]
        else:
            args2=args

        r = GeoJSON( *args2, **kwargs)
        return m.add_layer(r)
    m.add_GeoJSON = add_geojson 
    

    widgets.jslink((play, 'value'), (slider, 'value'))
    if number_of_images>1:
        return widgets.VBox([widgets.HBox([play,b2,b1,b3,b4, slider,label]),m])
    else:
        return widgets.VBox([widgets.HBox([b3,b4, label]),m])        
    #interactive(showim, i=slider   )
       

def geojson_lonlat_to_utm(aoi):
    """
    """
    # compute the utm zone number of the first polygon vertex
    lon, lat = aoi['coordinates'][0][0]
    utm_zone = utm.from_latlon(lat, lon)[2]

    # convert all polygon vertices coordinates from (lon, lat) to utm
    c = []
    for lon, lat in aoi['coordinates'][0]:
        c.append(utm.from_latlon(lat, lon, force_zone_number=utm_zone)[:2])

    return geojson.Polygon([c])


### DISPLAY IMAGES AND TABLES IN THE NOTEBOOK


# utility function for printing with Markdown format
def printmd(string):
    from IPython.display import Markdown, display
    display(Markdown(string))

    
def printbf(obj):
    printmd("__"+str(obj)+"__")

    
def show_array(a, fmt='jpeg'):
    ''' 
    display a numpy array as an image
    supports monochrome (shape = (N,M,1) or (N,M))
    and color arrays (N,M,3)  
    '''
    import PIL.Image
    from io import BytesIO
    import IPython.display
    import numpy as np
    f = BytesIO()
    PIL.Image.fromarray(np.uint8(a).squeeze() ).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


def display_image(img):
    '''
    display_image(img)
    display an image in the curren IPython notebook
    img can be an url, a local path, or a numpy array
    '''
    from IPython.display import display, Image
    from urllib import parse   
    import numpy as np
    
    if type(img) == np.ndarray:
        x = np.squeeze(img).copy()
        show_array(x)
    elif parse.urlparse(img).scheme in ('http', 'https', 'ftp'):
        display(Image(url=img)) 
    else:
        display(Image(filename=img)) 
        
        
def display_imshow(im,
                   range=None,
                   cmap='gray',
                   axis='equal',
                   cbar=False,
                   box=False,
                   title = None,
                   invert=False):
    '''
    display_imshow(img)
    display an numpy array using matplotlib
    img can be an url, a local path, or a numpy array
    range is a list [vmin, vmax]
    cmap sets the colormap ('gray', 'jet', ...)
    axis sets the scale of the axis ('auto', 'equal', 'off')
          https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.axis.html
    invert reverses the y-axis
    '''
    import matplotlib.pyplot as plt
    vmin,vmax=None,None
    if range:
        vmin,vmax = range[0],range[1]
    f, ax = plt.subplots()
    im = ax.imshow(im.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax)
    if invert:
        ax.gca().invert_yaxis()
    if cbar:
        f.colorbar(im, ax = ax)
    if title:
        ax.set_title(title)
    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.show() 
        

def display_table(table):
    import pandas as pd   # display tables
    from IPython.display import display
    display(pd.DataFrame(table))


def urlencoded_jpeg_img(a):
    ''' 
    returns the string of an html img tag with the urlencoded jpeg of 'a'
    supports monochrome (shape = (N,M,1) or (N,M))
    and color arrays (N,M,3)  
    '''
    fmt='jpeg'
    import PIL.Image
    from io import BytesIO
    import IPython.display
    import numpy as np
    f = BytesIO()
    import base64
    PIL.Image.fromarray(np.uint8(a).squeeze() ).save(f, fmt)
    x =  base64.b64encode(f.getvalue())
    return '''<img src="data:image/jpeg;base64,{}&#10;"/>'''.format(x.decode())
    # display using IPython.display.HTML(retval)
    
       
### initialize gallery
        
gallery_style_base = """
    <style>
.gallery2 {
    position: relative;
    width: auto;
    height: 450px; }
.gallery2 .index {
    padding: 0;
    margin: 0;
    width: 10.5em;
    list-style: none; }
.gallery2 .index li {
    margin: 0;
    padding: 0;
    float: left;}
.gallery2 .index a { /* gallery2 item title */
    display: block;
    background-color: #EEEEEE;
    border: 1px solid #FFFFFF;
    text-decoration: none;
    width: 1.9em;
    padding: 6px; }
.gallery2 .index a span { /* gallery2 item content */
    display: block;
    position: absolute;
    left: -9999px; /* hidden */
    top: 0em;
    padding-left: 0em; }
.gallery2 .index a span img{ /* gallery2 item content */
    width: 100%;
    }
.gallery2 .index li:first-child a span {
    top: 0em;
    left: 10.5em;
    z-index: 99; }
.gallery2 .index a:hover {
    border: 1px solid #888888; }
.gallery2 .index a:hover span {
    left: 10.5em;
    z-index: 100; }
</style>
    """

  
def display_gallery(image_urls, image_labels=None):
    '''
    image_urls can be a list of urls 
    or a list of numpy arrays
    image_labels is a list of strings
    '''
    from  IPython.display import HTML  
    import numpy as np

    
    gallery_template = """
    <div class="gallery2">
        <ul class="index">
            {}
        </ul>
    </div>
    """
    
    li_template = """<li><a href="#">{}<span style="background-color: white;  " ><img src="{}" />{}</span></a></li>"""
    li_template_encoded = """<li><a href="#">{}<span style="background-color: white;  " >{}{}</span></a></li>"""

    li = ""
    idx = 0
    for u in image_urls:
        if image_labels:
            label = image_labels[idx]
        else:
            label = str(idx)
        if type(u) == str:
            li = li + li_template.format( idx, u, label)
        elif type(u) == np.ndarray:
            li = li + li_template_encoded.format( idx, urlencoded_jpeg_img(u), label)

        idx = idx + 1
        
    source = gallery_template.format(li)
    
    display(HTML( source ))
    display(HTML( gallery_style_base ))

    return 
    

### READ AND WRITE IMAGES
    
    
    

import rasterio
import warnings

def rio_open(p):
    """
    Open an image with rasterio.

    Args:
        p: path to the image file

    Returns:
        rasterio dataset
    """
    with warnings.catch_warnings():  # noisy warning may occur here
        warnings.filterwarnings("ignore",
                                category=UserWarning)
        return rasterio.open(p)    
    

    
def readGTIFF(fname):
    '''
    Reads an image file into a numpy array, 
    returns the numpy array with dimensios (height, width, channels) 
    The returned numpy array is always of type numpy.float 
    '''
    import rasterio
    import numpy as np
    # TODO: suppress verbose warnings
    # read the image into a np.array 
    with  rio_open(fname) as s:
        # print('reading image of size: %s'%str(im.shape))
        im = s.read()
    return im.transpose([1,2,0]).astype(np.float)

  
def readGTIFFmeta(fname):
    '''
    Reads the image GeoTIFF metadata using rasterio and returns it,
    along with the bounding box, in a tuple: (meta, bounds)
    if the file format doesn't support metadata the returned metadata is invalid
    This is the metadata rasterio was capable to interpret, 
    but the ultimate command for reading metadata is *gdalinfo* 
    '''
    import rasterio
    import numpy as np
    # read the image into a np.array 
    with  rio_open(fname) as s:
        ## interesting information
        # print(s.crs,s.meta,s.bounds)
        return (s.meta,s.bounds)


def get_driver_from_extension(filename):
    import os.path
    ext = os.path.splitext(filename)[1].upper()
    if ext in ('.TIF', '.TIFF'):
        return 'GTiff'
    elif ext in ('.JPG', '.JPEG'):
        return 'JPEG'
    elif ext == '.PNG':
        return 'PNG'
    return None


def writeGTIFF(im, fname, copy_metadata_from=None):
    '''
    Writes a numpy array to a GeoTIFF, PNG, or JPEG image depending on fname extension.
    For GeoTIFF files the metadata can be copied from another file.
    Note that if  im  and  copy_metadata_from have different size, 
    the copied geolocation properties are not adapted. 
    '''
    import rasterio
    import numpy as np

    # set default metadata profile
    p = {'width': 0, 'height': 0, 'count': 1, 'dtype': 'uint8', 'driver': 'PNG',
             'affine': rasterio.Affine (0,1,0,0,1,0),
             'crs': rasterio.crs.CRS({'init': 'epsg:32610'}), 
             'tiled': False,  'nodata': None}
    
    # read and update input metadata if available 
    if copy_metadata_from:   
        x = rio_open(copy_metadata_from)
        p.update( x.profile )    
 
    # format input 
    if  len(im.shape) == 2: 
        im = im[:,:,np.newaxis]
        
    # override driver and shape
    indriver = get_driver_from_extension(fname)
    if indriver and (indriver != p['driver']):
        print('writeGTIFF: driver override from %s to %s'%( p['driver'], indriver))
        p['driver'] = indriver or p['driver']
    
    if indriver == 'GTiff' and (p['height'] != im.shape[0]  or  p['width'] != im.shape[1]):
        # this is a problem only for GTiff
        print('writeGTIFF: changing the size of the GeoTIFF')
    else:     
        # remove useless properties 
        p.pop('tiled')
    
    p['height'] = im.shape[0]
    p['width']  = im.shape[1]
    p['count']  = im.shape[2]

    with rasterio.open(fname, 'w', **p) as d:
            d.write((im.transpose([2,0,1]).astype(d.profile['dtype'])))


def overprintText(im,imout,text,textRGBA=(255,255,255,255)):
    '''
    prints text in the upper left corner of im (filename) 
    and writes imout (filename)
    '''
    from PIL import Image, ImageDraw, ImageFont
    # get an image
    base = Image.open(im).convert('RGBA')

    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', base.size, (255,255,255,0))

    # get a font
    #    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    # get a drawing context
    d = ImageDraw.Draw(txt)

    # draw text
    d.text((1,1), text,  fill=tuple(textRGBA))
    out = Image.alpha_composite(base, txt)

    out.save(imout)


def epsgcode_to_utmcode(epsgcode):
    # https://gis.stackexchange.com/questions/20298/is-it-possible-to-get-the-epsg-value-from-an-osr-spatialreference-class-using-th
    # returns the UTM zone, and a boolean northen 
    from osgeo import gdal,osr
    import re
    srs = osr.SpatialReference()
    srs.SetFromUserInput(epsgcode)
    #print(srs)
    UTMzoneNS = srs.GetAttrValue("PROJCS", 0).split(' ')[-1]
    print (UTMzoneNS)
    _, UTMzone, NS, _ = re.split('(\d+)([a-zA-Z])',UTMzoneNS)
    UTMzone = int(UTMzone)
    NS = (NS == 'N')
    return UTMzone, NS


#### wrappers for gdaltransform and gdalwarp

       
def gdal_get_longlat_of_pixel(fname, x, y, verbose=True):
    '''
    returns the longitude latitude and altitude (wrt the WGS84 reference 
    ellipsoid) for the points at pixel coordinates (x, y) of the image fname. 
    The CRS of the input GeoTIFF is determined from the metadata in the file.

    '''
    import subprocess

    # form the query string for gdaltransform
    q = b''
    for (xi,yi) in zip(x,y):
        q = q + b'%d %d\n'%(xi,yi)
    # call gdaltransform, "+proj=longlat" uses the WGS84 ellipsoid 
    #    echo '0 0' | gdaltransform -t_srs "+proj=longlat" inputimage.tif 
    cmdlist = ['gdaltransform', '-t_srs', "+proj=longlat", fname]
    if verbose:
        print ('RUN: ' +  ' '.join(cmdlist) + ' [x y from stdin]')
    p = subprocess.Popen(cmdlist,
                         stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    out = (p.communicate(q)[0]).decode()
    listeout =  [ list(map(float, x.split())) for x in out.splitlines()]
    return listeout


def gdal_resample_image_to_longlat(fname, outfname, verbose=True):
    '''
    resample a geotiff image file in longlat coordinates (EPSG: 4326 with WGS84 datum)
    and saves the result in outfname
    '''
    import os
    
    driver = get_driver_from_extension(outfname)
    cmd = 'gdalwarp -overwrite  -of %s -t_srs "+proj=longlat +datum=WGS84" %s %s'%(driver, fname, outfname)
    if verbose:
        print('RUN: ' + cmd)
    return os.system(cmd)

def display_RSO(img, title = None, cbar = False, plot = True):
    """
    Display an RSO image

    Input:
    img:     numpy array
             raw image
    [title]: string, optional
             title for display_imshow call

    Output:
    -------
    clipped_image
    """
    import numpy as np
    assert type(img) == np.ndarray, "Input is not a numpy array"
    new = np.copy(np.abs(img))
    np.clip(new, 0, new.mean() + 3 * new.std(), out = new)
    new = 255 * (new - new.min()) / (new.max() - new.min())
    if plot == True:
        display_imshow(new, title = title, cbar = cbar)
    return new

def simple_equalization_8bit(im, percentiles=5):
    ''' im is a numpy array
        returns a numpy array
    '''
    import numpy as np
    mi, ma = np.nanpercentile(im.flatten(), (percentiles,100-percentiles))
    im = np.minimum(np.maximum(im,mi), ma) # clip
    im = (im-mi)/(ma-mi)*255.0   # scale
    im=im.astype(np.uint8)
    return im