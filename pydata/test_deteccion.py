import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from skimage import feature
from skimage.io import imread
from skimage import filters
from skimage.measure import regionprops, label
from circle_fit import taubinSVD

image = imread('prueba.bmp').astype('double')



def deteccion(image):

    imt = image < filters.threshold_otsu(image)/2


    # Compute the Canny filter for two values of sigma
    edges = feature.canny(imt, sigma=5)
    L = label(edges)

    props = regionprops(L)

    # Calcular función de costo y guardar (evitando divisiones por cero)
    props_with_cost = []
    for region in props:
        major = region.major_axis_length
        minor = region.minor_axis_length
        if major > 0:  # evitar división por cero
            cost = region.area * (minor / major)
            props_with_cost.append((region.label, cost))

    # Ordenar por costo de mayor a menor
    props_sorted_by_cost = sorted(props_with_cost, key=lambda x: x[1], reverse=True)

    # Obtener los dos con mayor costo
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

    return ycma, xcma, rma, ycmi, xcmi, rmi


ycma, xcma, rma, ycmi, xcmi, rmi = deteccion(image)


theta = np.linspace(0, 2*np.pi, 1000)

plt.figure()
plt.imshow(image)
plt.plot(xcma+rma*np.cos(theta), ycma+rma*np.sin(theta), 'r.-')
plt.plot(xcmi+rmi*np.cos(theta), ycmi+rmi*np.sin(theta), 'r.-')
