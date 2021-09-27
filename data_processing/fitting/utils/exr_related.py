import OpenEXR, Imath, array
import numpy as np

def read_exr(exr_path):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(exr_path)
    dw = golden.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = golden.channel('R', pt)
    red = np.fromstring(redstr, dtype = np.float32)
    red.shape = (size[1], size[0],1) # Numpy arrays are (row, col)

    greenstr = golden.channel('G', pt)
    green = np.fromstring(greenstr, dtype = np.float32)
    green.shape = (size[1], size[0],1) # Numpy arrays are (row, col)

    bluestr = golden.channel('B', pt)
    blue = np.fromstring(bluestr, dtype = np.float32)
    blue.shape = (size[1], size[0],1) # Numpy arrays are (row, col)
    
    img = np.concatenate([red,green,blue],axis=-1)

    return img