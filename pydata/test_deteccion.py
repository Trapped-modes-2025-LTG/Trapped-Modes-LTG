import numpy as np
#!pip install circle_fit
import matplotlib.pyplot as plt
plt.ion()
import circle_fit
from skimage import feature
from skimage.io import imread
from skimage import filters
from skimage.measure import regionprops, label
from circle_fit import taubinSVD

#%%
#image = imread('structure.bmp').astype('double')

#%%

#image = imread('/Users/mateoeljatib/Documents/structure.bmp').astype(np.double)


#%%

def deteccion(image):

    imt = image < filters.threshold_otsu(image)/2


    # Compute the Canny filter for two values of sigma
    edges = feature.canny(imt, sigma = 10) #TODO: change sigma if needed 
    L = label(edges)

    props = regionprops(L)

    # Calculate cost function
    props_with_cost = []
    for region in props:
        major = region.major_axis_length
        minor = region.minor_axis_length
        if major > 0:  # evitar división por cero
            cost = region.area * (minor / major)
            props_with_cost.append((region.label, cost))

    # sorting by cost 
    props_sorted_by_cost = sorted(props_with_cost, key=lambda x: x[1], reverse=True)

    # the two wit higher cost (for I'm seeking for 2 circles)
    top_two_labels = [item[0] for item in props_sorted_by_cost[:2]]

    XYma = props[top_two_labels[0]-1].coords
    XYmi = props[top_two_labels[1]-1].coords

    # plt.figure()
    # plt.imshow(image)
    # plt.plot(XYma[:,1], XYma[:,0], 'r.')
    # plt.plot(XYmi[:,1], XYmi[:,0], 'r.')
    # plt.colorbar()



    ycma, xcma, rma, sigmama = taubinSVD(XYma)
    ycmi, xcmi, rmi, sigmimi = taubinSVD(XYmi)
    ###
    ny, nx = image.shape
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    # Distance of each pixel to the center of each circle
    dist_to_big = np.sqrt((Y - ycma)**2 + (X - xcma)**2)
    dist_to_small = np.sqrt((Y - ycmi)**2 + (X - xcmi)**2)

    # Mask to keep only what's inside the small cricle
    # or outside the big one 
    mask = (dist_to_big > rma) | (dist_to_small < rmi)

    # applying mask to the image
    masked_image = image.copy()
    masked_image[~mask] = 0  
    
    return masked_image, ycma, xcma, rma, ycmi, xcmi, rmi



