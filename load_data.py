import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

import PIL
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from circle_fit import leastsq_circle
from skimage.draw import line
from skimage.transform import resize
from cv2 import equalizeHist, createCLAHE
import random
import math
from skimage.measure import subdivide_polygon
import pickle
Image.MAX_IMAGE_PIXELS = 1000000000



class Rim(object):
    def __init__(self, coordinates, img, mosaic=None):
        self.mosaic = mosaic
        self.cropped = None
        self.target = None
        self.top = img.size[1]
        self.edge = img.size[0]
        self.coords = []
        for coord in coordinates:
            coords = [float(x) for x in coord.split('\t')]
            coords[1] = self.top - coords[1]
            coords = tuple(coords)
            self.coords.append(coords)
        X0 = np.array([x[0] for x in self.coords])
        X1 = np.array([x[1] for x in self.coords])
        if any(X0 < 0) or any(X0 > self.edge):
            raise ValueError('Trace point outside image boundaries')
        if any(X1 < 0) or any(X1 > self.top):
            raise ValueError('Trace point outside image boundaries')
        prev_crd = self.coords[-1]
        self.d = []
        for i, crd in enumerate(self.coords):
            d = np.sqrt(np.sum(np.square(np.array(crd) - np.array(prev_crd))))
            self.d.append(d)
            prev_crd = crd
        self.min_0 = min([x[0] for x in self.coords])
        self.min_1 = min([x[1] for x in self.coords])
        self.fit_circle()
        self.create_footprint(img)
        
    def fit_circle(self):
        """Fits a circle to the points on rim."""
        X0 = [x[0] for x in self.coords]
        X1 = [x[1] for x in self.coords]
        xc, yc, R, residu = leastsq_circle(X0, X1)
        self.c0 = xc
        self.c1 = yc
        self.r = R
        self.residual = residu
        self.res_ratio = self.residual/self.r
            
    def show(self):
        """Plots trace points in scatter plot."""
        arr = np.array(self.cropped)
        plt.figure(figsize=(10, 10))
        plt.imshow(arr, cmap='Greys_r');
        plt.scatter([z[1] for z in self.coords], [z[0] for z in self.coords])
        plt.show();
        
    def create_footprint(self, img, in_dim=224, scale_factor=3):
        scale_factor = 3
        scale = self.r * scale_factor * 2
        left = self.c0 - scale
        upper = self.c1 - scale
        right = self.c0 + scale
        lower = self.c1 + scale
        cropped = img.crop((left, upper, right, lower))
        dim = cropped.size[0]
        target = np.zeros((dim, dim))
        mean_d = np.array(self.d).mean()
        coords = []
        for coord in self.coords:
            crd = (coord[1] - self.c1 + scale, coord[0] - self.c0 + scale)
            coords.append(crd)
        self.coords = coords
        self.cropped = cropped
        return
        
    def create_target(self, img, in_dim=224, scale_factor=3):
        scale_factor = 3
        scale = self.r * scale_factor * 2
        left = self.c0 - scale
        upper = self.c1 - scale
        right = self.c0 + scale
        lower = self.c1 + scale
        cropped = img.crop((left, upper, right, lower))
        dim = cropped.size[0]
        target = np.zeros((dim, dim))
        mean_d = np.array(self.d).mean()
        for i, d in enumerate(self.d):
            if d < mean_d * 2.5:
                ln = line(
                    int(round(self.coords[i-1][1] - self.c1 + scale)), 
                    int(round(self.coords[i-1][0] - self.c0 + scale)), 
                    int(round(self.coords[i][1] - self.c1 + scale)), 
                    int(round(self.coords[i][0] - self.c0 + scale))
                )
                target[ln] = 255
        in_dim *= 2
        in_dim += 5
        in_dim = int(in_dim)
        target_image = Image.fromarray(np.uint8(target))
        cropped = cropped.resize((in_dim, in_dim), resample=PIL.Image.BILINEAR)
        target_image = target_image.resize((in_dim, in_dim), resample=PIL.Image.BILINEAR)
        self.cropped = cropped
        self.target = target_image
        return
    
    def get_pair(self, out_dim = 224, rotation=0, displace=(0,0), rescale=1):
        if self.cropped == None:
            raise Exception('No defined image/target pair')
        dim = int(out_dim*rescale)
        buff = (self.cropped.size[0] - dim)//2
        left = buff + displace[0]
        right = buff + dim + displace[0]
        top = buff + displace[1]
        bottom = buff + dim + displace[1]
        to_crop = self.cropped.rotate(rotation)
        target = self.target.rotate(rotation)
        img = to_crop.crop((left, top, right, bottom))
        target = target.crop((left, top, right, bottom))
        if img.size[0] != out_dim:
            img = img.resize((out_dim, out_dim), resample=PIL.Image.BILINEAR)
            target = target.resize((out_dim, out_dim), resample=PIL.Image.BILINEAR)
        return img, target
    
    def rotate_around_point(self, xy, radians, origin=(0, 0)):
        """Rotate a point around a given point.
        Credit: Lyle Scott
        https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
        """
        x, y = xy
        offset_x, offset_y = origin
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = math.cos(radians)
        sin_rad = math.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        return qx, qy
    
    def draw_rim(self, crop_off=.25, res=224, rot=0, disp=(0,0)):
        img = self.cropped
        dim = self.cropped.size[0]
        coords = self.coords
        out = []
        for pnt in coords:
            point = (pnt[1], pnt[0])
            as_radians = rot*math.pi/180
            point = self.rotate_around_point(point, as_radians, origin=(dim/2, dim/2))
            out.append((point[1], point[0]))
        coords = out
        img = img.rotate(rot)
        crop_off = int(crop_off*img.size[0])
        img_post_size = img.size[0]-(2*crop_off)
        res_factor = res/img_post_size
        disp0 = disp[0] / res_factor
        disp1 = disp[1] / res_factor
        img = img.crop((crop_off - disp1, crop_off-disp0, img.size[0]-crop_off-disp1, img.size[1]-crop_off-disp0))
        img = img.resize((res, res), resample=PIL.Image.BILINEAR)
        arr = np.array(img)
        coords = [(((x[0]-crop_off) * res_factor)+disp[0], ((x[1]-crop_off) * res_factor)+disp[1]) for x in coords]
        target = np.zeros((res, res))
        coords = np.array([list(x) for x in coords])
        coord_groups=[]
        mean_d = np.mean(self.d)
        std = np.std(self.d)
        crds = []
        thresh = 2.5
        for i, d in enumerate(self.d):
            if i == 0:
                if d < thresh*mean_d:
                    crds.append(coords[-1])
            if d < thresh*mean_d:
                crds.append(coords[i])
            elif len(crds) > 1:
                crds = np.array([list(x) for x in crds])
                coord_groups.append(crds)
                crds = []
            else:
                pass
        crds = np.array([list(x) for x in crds])
        coord_groups.append(crds)
        for crds in coord_groups:
            new_coords = crds.copy()
            for _ in range(5):
                new_coords = subdivide_polygon(new_coords, degree=2, preserve_ends=True)
            crds = new_coords
            rounded = set()
            for crd in crds:
                nxt = (int(round(crd[0])), int(round(crd[1])))
                rounded.add(nxt)
            pxls = (np.array([x[0] for x in rounded]), np.array([x[1] for x in rounded]))
            target[pxls] = 255
        return arr, target
    
def load_craters(directory='./data/pickles/test/'):
    """Loads the data from the specified directory."""
    rims = []
    files = os.listdir(directory)
    files = [directory + x for x in files]
    for fl in files:
        with open(fl, 'rb') as f:
            rims.append(pickle.load(f))
    return rims

def random_in_range(lower, upper):
    center = (lower + upper)/2
    rng = upper - lower
    val = random.random() - .5
    out = val*rng + center
    return out

def get_thresh(radius):
    """Computes threshold for binarization."""
    if radius < 10:
        return .5
    if radius < 100:
        return 4/radius
    else:
        return 180/radius**2
    
def make_channels(input_image):
    """Converts image to three channels:
    1: unaltered image
    2: Histogram equalized image
    3: CLAHE image
    """
    ch1 = np.array(input_image)
    ch2 = equalizeHist(ch1)
    clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ch3 = clahe.apply(ch1)
    image = [ch1, ch2, ch3]
    image = [np.expand_dims(x, axis=-1) for x in image]
    image = np.concatenate(image, axis=-1)
    return image


def datagen(rims, val=False, batch_size=8):
    """Generates training batches."""
    X = []
    Y = []
    count = 0
    while True:
        if len(X) == batch_size:
            X = [np.expand_dims(x, axis=0) for x in X]
            X = np.concatenate(X, axis=0)
            Y = [np.expand_dims(y, axis=0) for y in Y]
            Y = np.concatenate(Y, axis=0)
            yield X, Y
            X = []
            Y = []
        else:
            if val:
                if count == len(rims):
                    count = 0
                rim = rims[count]
                image, target = rim.get_pair()
                count += 1
            elif not val:
                rim = random.choice(rims)
                rot = random.randint(0, 360)
                disp0 = random.randint(-5, 5)
                disp1 = random.randint(-5, 5)
                rescale = random_in_range(.5, 1.5)
                image, target = rim.get_pair(
                    rotation=rot, 
                    displace=(disp0, disp1),
                    rescale=rescale
                )
            image = make_channels(image)/255
            target = np.array(target)/255
            thresh = get_thresh(rim.r)
            target = np.where(target > thresh, 1, 0)
            X.append(image)
            Y.append(target)
            


def prepare_image(rim, rot=0, disp=(0,0), rescale=1):
    """Generates training batches."""
    X = []
    Y = []
    disp0 = disp[0]
    disp1 = disp[1]
    image, target = rim.get_pair(
        rotation=rot, 
        displace=(disp0, disp1),
        rescale=rescale
    )
    image = make_channels(image)/255
    target = np.array(target)/255
    thresh = get_thresh(rim.r)
    target = np.where(target > thresh, 1, 0)
    X.append(image)
    Y.append(target)
    return np.array(X), np.array(Y)


    
    