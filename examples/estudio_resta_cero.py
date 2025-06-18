import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze
from skimage.draw import polygon
from skimage.measure import label, regionprops
#%% Cargar carpeta de mediciones ancladas

base_dir = os.path.dirname(__file__)

map_folder = os.path.join(base_dir, "Pictures", "29_05", "maps")

calibration_files = [f for f in os.listdir(map_folder) if 'calibration_factor' in f and f.endswith('.npy')]

calibration_path = os.path.join(map_folder, calibration_files[0])
calibration_factor = np.load(calibration_path)

file_list = sorted([f for f in os.listdir(map_folder) if f.endswith('_map.npy') and 'calibration_factor' not in f])


#%%


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



evolucion_punto = cargar_centros(map_folder, image_shape)
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
'''
nuevo nuevo nuevooo
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

#%% Paths
base_dir = os.path.dirname(__file__)
map_folder = os.path.join(base_dir, "Pictures", "29_05", "maps")

file_list = sorted([f for f in os.listdir(map_folder) if f.endswith('_map.npy') and 'calibration_factor' not in f])
map_paths = [os.path.join(map_folder, f) for f in file_list]

#%% Cargar mapas y encontrar punto válido
primer_mapa = np.load(map_paths[0])
image_shape = primer_mapa.shape

# Encontrar punto válido aleatorio (no NaN, no 0)
valid_indices = np.argwhere((~np.isnan(primer_mapa)) & (primer_mapa != 0))
if valid_indices.size == 0:
    raise ValueError("No se encontraron puntos válidos en el primer mapa.")

np.random.seed(2)  # reproducibilidad
iy, ix = valid_indices[np.random.choice(len(valid_indices))]

print(f"Punto elegido: ({iy}, {ix})")
# Mostrar visualización del punto elegido
plt.figure(figsize=(6, 6))
plt.imshow(primer_mapa, cmap='viridis')
plt.colorbar(label='Altura')
plt.scatter(ix, iy, color='red', s=80, label=f'Punto elegido ({iy}, {ix})')
plt.title("Primer mapa de altura con punto elegido")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

#%% Obtener evolución temporal de ese punto
alturas = []
for path in map_paths:
    mapa = np.load(path)
    val = mapa[iy, ix]
    alturas.append(val)

alturas = np.array(alturas)
tiempos = np.arange(len(alturas))*1/500  # puedes reemplazar por los tiempos reales si los tenés

#%% Transformada de Fourier y análisis armónico
N = len(alturas)
dt = 1/500 

fft_vals = fft(alturas)
fft_vals = fft_vals / N
frecuencias = fftfreq(N, d=dt)

# Usar solo positivas
frecuencias_pos = frecuencias[:N//2]
fft_pos = fft_vals[:N//2]

fft_mod = np.abs(fft_pos)
peaks, _ = find_peaks(fft_mod)

if len(peaks) == 0:
    raise ValueError("No se encontraron picos en la transformada.")

max_peak_index = peaks[np.argmax(fft_mod[peaks])]
f0 = frecuencias_pos[max_peak_index]
print(f"Frecuencia fundamental f0: {f0}")

harmonics = [f0 * n for n in range(0, 4)]
indices = [np.argmin(np.abs(frecuencias_pos - f)) for f in harmonics]

amps = []
phases = []

for i, idx in enumerate(indices):
    coeff = fft_pos[idx]
    if i == 0:
        A_n = np.abs(coeff)  # modo DC no se multiplica por 2
    else:
        A_n = np.abs(coeff) * 2
    phi_n = np.angle(coeff)
    amps.append(A_n)
    phases.append(phi_n)


#%% Reconstrucción con modos armónicos
plt.figure()
plt.plot(tiempos, alturas, 'k', label='Original')

A0 = amps[0]  # DC
armonics = [A0 * np.ones_like(tiempos)]

for n, (A_n, phi_n) in enumerate(zip(amps[1:], phases[1:]), start=1):
    armonic = A_n * np.cos(2 * np.pi * f0 * n * tiempos + phi_n)
    armonics.append(armonic)

for i, armonic in enumerate(armonics):
    plt.plot(tiempos, armonic, label=f'n = {i}')

plt.legend()
plt.title("Descomposición armónica de la altura")
plt.xlabel("Tiempo")
plt.ylabel(f"Altura en el punto seleccionado ({iy}, {ix})")
plt.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

#%%
# Suma progresiva de armónicos
suma = np.zeros_like(tiempos)
plt.figure()

for i, armonic in enumerate(armonics):
    suma += armonic
    plt.plot(tiempos, suma, label=f'Suma hasta n = {i}')

plt.plot(tiempos, alturas, 'k--', linewidth=2, label='Original')
plt.legend()
plt.title("Reconstrucción progresiva de la señal")
plt.xlabel("Tiempo")
plt.ylabel("Altura en el punto")
plt.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

#%%

# Extraemos el modo cero (DC)
A0 = amps[0]  # ya lo tenés calculado

# Restamos el modo DC a la señal original
alturas_centrada = alturas - A0

# Graficar para comparar
plt.figure(figsize=(10,5))
plt.plot(tiempos, alturas, label='Original')
plt.plot(tiempos, alturas_centrada, label='Señal sin modo DC')
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.title('Señal original y señal con modo cero restado')
plt.xlabel('Tiempo')
plt.ylabel('Altura en el punto seleccionado')
plt.grid(True)
plt.tight_layout()
plt.show()


print(f"A0 (DC): {amps[0]:.6f} vs promedio real: {np.mean(alturas):.6f}")


#%%
from tqdm import tqdm

#%% Cargar todos los mapas
file_list = sorted([f for f in os.listdir(map_folder) if f.endswith('_map.npy') and 'calibration_factor' not in f])
map_paths = [os.path.join(map_folder, f) for f in file_list]

# Cargar todos los mapas en un array 3D: (N_frames, H, W)
stack = []
for path in map_paths:
    mapa = np.load(path)
    stack.append(mapa)
stack = np.array(stack)

N, H, W = stack.shape
dt = 1 / 500

#%% Inicializar matrices de salida
modo_DC = np.full((H, W), np.nan)
media_temporal = np.full((H, W), np.nan)

#%% Recorrer cada píxel con barra de progreso
for i in tqdm(range(H), desc="Procesando filas"):
    for j in range(W):
        señal = stack[:, i, j]  # señal temporal en (i,j)

        # Evitar píxeles inválidos (NaN o todos ceros)
        if np.any(np.isnan(señal)) or np.all(señal == 0):
            continue

        # Media temporal
        media_temporal[i, j] = np.mean(señal)

        # FFT y modo DC
        fft_vals = fft(señal) / N
        modo_DC[i, j] = np.abs(fft_vals[0])

#%% Ver diferencia entre media y modo DC
diferencia = (media_temporal - modo_DC)/np.nanmean(modo_DC)

#%% Graficar (opcional, si usás matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(media_temporal, cmap='twilight')
plt.title("Media temporal")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(modo_DC, cmap='twilight')
plt.title("Modo DC (FFT)")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(-diferencia, cmap='twilight', vmin=-np.nanmax(np.abs(diferencia)), vmax=np.nanmax(np.abs(diferencia)))
plt.title("(Media - Modo DC )/ Modo DC ")
plt.colorbar()

plt.tight_layout()
plt.show()

#%%

diff = np.mean(diferencia**2)
