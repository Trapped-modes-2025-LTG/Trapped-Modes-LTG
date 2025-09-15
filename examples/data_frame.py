import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze
from skimage.draw import polygon
from skimage.measure import label, regionprops
#%%
base_dir = os.path.dirname(__file__)

tif_folder = os.path.join(base_dir, "Pictures", "mask", "29_05")

reference_path = os.path.join(base_dir, "Pictures", "mask", "29_05", "reference.tif")

#%% Chequear parámetros de la máscara

first_tif_path = None
with os.scandir(tif_folder) as entries:
    for entry in entries:
        if entry.name.endswith(".tif") and entry.is_file():
            first_tif_path = entry.path
            break
        
image_for_mask = analyze.load_image(first_tif_path)        
        
mask, contornos = analyze.mask(image_for_mask,
                    smoothed = 15, 
                    percentage = 95,
                    show_mask = True
                    )       

#%% Folder

layers = [[3.2e-2,1.0003], [ 1.2e-2,1.48899], [3.4e-2,1.34], [ 80e-2 ,1.0003]]

analyze.folder(reference_path, tif_folder, layers, 0.002, 
               smoothed=15, percentage =95 ,timer= True, only="both")



#%%

contours_folder = os.path.join(base_dir, "Pictures", "mask", "29_05", "maps")
image_shape = (1024, 1024)  # reemplazá por el tamaño real de tus imágenes
puntos = []
def cargar_centros(contours_dir, image_shape):
    files = sorted([f for f in os.listdir(contours_dir) if f.endswith("_contours.npy")])
    total = len(files)

    for i, fname in enumerate(files, 1):  # el 1 es para que empiece desde 1
        print(f"Procesando archivo {i} de {total}: {fname}")
        
        path = os.path.join(contours_dir, fname)
        contornos = np.load(path, allow_pickle=True)

        if len(contornos) < 2:
            print(f"  -> Menos de dos contornos, se omite.")
            continue

        contornos = sorted(contornos, key=lambda c: len(c), reverse=True)
        cnt = contornos[1]

        mask = np.zeros(image_shape, dtype=np.uint8)
        r = cnt[:, 1]
        c = cnt[:, 0]
        rr, cc = polygon(r, c, shape=mask.shape)
        mask[rr, cc] = 1

        labeled = label(mask)
        props = regionprops(labeled)
        if props:
            cy, cx = props[0].centroid
            puntos.append((int(round(cy)), int(round(cx))))
        else:
            puntos.append((np.nan, np.nan))
            print("  -> No se pudo encontrar centroide.")

    print(f"\nProcesamiento completo: {len(puntos)} centros calculados.")
    return np.array(puntos)



evolucion_punto = cargar_centros(contours_folder, image_shape)

#%%
test = np.linspace(0, len(puntos), len(puntos))

plt.plot(test, puntos)
plt.title("Posición en X y en Y del centro")
plt.show()

#%%
puntos_x = [p[0] for p in puntos]
puntos_y = [p[1] for p in puntos]

plt.plot(test, puntos_y, label='Coordenada Y del centro')
plt.plot(test, puntos_x, label='Coordenada X del centro')
plt.title("Posición en X y Y del centro a lo largo del tiempo")
plt.xlabel("Tiempo (índice o segundos)")  # ajustá si test es tiempo real
plt.ylabel("Posición (pixeles)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

calibration_files = [f for f in os.listdir(contours_folder) if 'calibration_factor' in f and f.endswith('.npy')]
if not calibration_files:
    raise FileNotFoundError("No se encontró el archivo de calibration factor en la carpeta.")

calibration_path = os.path.join(contours_folder, calibration_files[0])
calibration_factor = np.load(calibration_path)
print("Calibration factor encontrado:", calibration_factor)


file_list = sorted([f for f in os.listdir(contours_folder) if f.endswith('_map.npy') and 'calibration_factor' not in f])


tiempos = [i * (1 / 500) for i in range(len(file_list))]

# data = [
#     calibration_factor * np.load(os.path.join(contours_folder, f))
#     for f in file_list
# ]
                                                                            # CHEQUEAR CALFACT
data = [ np.load(os.path.join(contours_folder, f))
    for f in file_list
]

df = pd.DataFrame({'tiempo': tiempos, 'data': data})
#%%

matriz = df.loc[574, 'data'] # pruebo en un frame random 

plt.imshow(matriz, cmap='viridis') 
plt.colorbar(label='Altura (calibrada)')
plt.title(f"Matriz en t = {df.loc[0, 'tiempo']:.3f} s")
plt.tight_layout()
plt.show()

#%%

matriz = df.loc[573, 'data'] 
x, y = puntos[0]  # Recordá: (fila, columna) 

plt.imshow(matriz, cmap='viridis')
plt.plot(x, y, 'ro')  # x en eje horizontal, y en vertical
plt.colorbar(label='Altura (calibrada)')
plt.title(f"Matriz en t = {df.loc[573, 'tiempo']:.3f} s")
plt.tight_layout()
plt.show()


#%%

puntos_int = np.round(puntos).astype(int)

# Invertimos columna y fila para acceder a frame[y, x] como frame[fila, columna]
evolucion = np.array([
    frame[col, row]  # invertido a propósito
    for frame, (row, col) in zip(df['data'], puntos_int)
])



#%%

plt.plot(test / 500, evolucion, label="Altura en el centro")  # dividís por 500 si eso da el tiempo en segundos
plt.title('Evolución temporal del centro del círculo')
plt.xlabel("Tiempo (s)")
plt.ylabel("Altura (calibrada)")
#plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# ESPECTROGRAMA DEL CENTRO

from scipy.signal import spectrogram

# df['tiempo']  --> array de tiempos
# evolucion --> señal 1D del valor en (i(t), j(t)) en el tiempo

# Parámetros de la señal
fs = 500  # frecuencia de muestreo en Hz (ya que haces i * 1/500)

# Espectrograma
nperseg =128
f, t, Sxx = spectrogram(evolucion, fs=fs, nperseg=nperseg, noverlap= 3*nperseg // 4)

#Visualización
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
plt.colorbar(label='Densidad espectral de potencia (dB/Hz)')
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')
plt.title("Espectrograma del centro")
plt.tight_layout()
plt.show()


#%%
contours_folder = os.path.join(base_dir, "Pictures", "mask", "29_05", "maps")
#%%
analyze.video(contours_folder)

