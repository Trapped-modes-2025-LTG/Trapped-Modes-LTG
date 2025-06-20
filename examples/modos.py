import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pydata.analyze import analyze
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp_polar
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

#%% Cargar carpeta de mediciones ancladas

base_dir = os.path.dirname(__file__)

map_folder = os.path.join(base_dir, "Pictures", "29_05", "maps")

calibration_files = [f for f in os.listdir(map_folder) if 'calibration_factor' in f and f.endswith('.npy')]

calibration_path = os.path.join(map_folder, calibration_files[0])
calibration_factor = np.load(calibration_path)

file_list = sorted([f for f in os.listdir(map_folder) if f.endswith('_map.npy') and 'calibration_factor' not in f])
#%%

num_blocks = 64
blocks_per_row = int(np.sqrt(num_blocks))
img_size = 1024
block_size = img_size // blocks_per_row
n_modes = 4

#%% Cargar modo 0 para obtener CERO
amp0_list = []
phase0_list = []

for idx in range(num_blocks):
    _, a, p = analyze.block_amplitude(map_folder, block_index=idx, mode=1)
    amp0_list.append(a[:, :, 0])
    phase0_list.append(p[:, :, 0])
    print(f"Modo 0 - bloque {idx+1}/{num_blocks}")

# Convertir a arrays completos
amp0_imgs = np.zeros((img_size, img_size))
phase0_imgs = np.zeros((img_size, img_size))

for idx in range(num_blocks):
    i = idx // blocks_per_row
    j = idx % blocks_per_row
    amp0_imgs[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = amp0_list[idx]
    phase0_imgs[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = phase0_list[idx]
# Construir matriz CERO (modo 0 espacial completo)
CERO = amp0_imgs * np.cos(phase0_imgs)

#%% Calcular modos 1 al n_modes con resta del modo 0
amp = []
phase = []

for idx in range(num_blocks):
    # i = idx // blocks_per_row
    # j = idx % blocks_per_row

    # Extraer bloque correspondiente de CERO
    #block_cero = CERO[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]

    _, a, p = analyze.block_amplitude(
        map_folder,
        block_index=idx,
        mode=n_modes,
        zero=0)#CERO)
    
    amp.append(a[:, :])
    phase.append(p[:, :])
    print(f"Modo {n_modes} - bloque {idx+1}/{num_blocks}")

#%% Reconstruir imágenes completas por modo
amp_imgs = [np.zeros((img_size, img_size)) for _ in range(n_modes)]
phase_imgs = [np.zeros((img_size, img_size)) for _ in range(n_modes)]

for idx in range(num_blocks):
    i = idx // blocks_per_row
    j = idx % blocks_per_row
    for mode in range(n_modes):
        amp_imgs[mode][i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = amp[idx][:,:,mode]
        phase_imgs[mode][i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = phase[idx][:,:,mode]

#%% Ploteo
fig, axes = plt.subplots(nrows=n_modes, ncols=2, figsize=(12, 4 * n_modes))

for mode in range(n_modes):
    ax_amp = axes[mode, 0]
    im_amp = ax_amp.imshow(amp_imgs[mode], cmap='inferno')
    ax_amp.set_title(f'Amplitud - Modo {mode}', fontsize=10)
    cbar = fig.colorbar(im_amp, ax=ax_amp)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.offsetText.set_fontsize(8)

    ax_phase = axes[mode, 1]
    im_phase = ax_phase.imshow(phase_imgs[mode], cmap='twilight')
    ax_phase.set_title(f'Fase - Modo {mode}', fontsize=10)
    cbar = fig.colorbar(im_phase, ax=ax_phase)
    cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()

#%%

UNO = amp_imgs[1]
DOS = amp_imgs[2]
TRES = amp_imgs[3]


#%%


contours_file = sorted([f for f in os.listdir(map_folder) if f.endswith('_contours.npy') and 'calibration_factor' not in f])

cont = contours_file[0]
#%%
foto = os.path.join(base_dir, "Pictures", "29_05","0_5mm_circular_100ms_20250529_141304_C1S0001000380.tif")
cy, cx = analyze.confined_peaks(analyze.load_image(foto), smoothed = 15, percentage= 90)

plt.scatter(cy, cx, s=20, color='red')
#%%
centro1 = tuple(map(int, (cx, cy)))

# Aplicar transformaciones polares (ya que el centro es el mismo)
polar1 = warp_polar(UNO, center=centro1, scaling='linear')
polar2 = warp_polar(DOS, center=centro1, scaling='linear')
polar3 = warp_polar(TRES, center=centro1, scaling='linear')

pnan1 = np.where(polar1 != 0, polar1, np.nan)
pnan2 = np.where(polar2 != 0, polar2, np.nan)
pnan3 = np.where(polar3 != 0, polar3, np.nan)

# Crear figura con 3 filas y 2 columnas
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Lista de imágenes y sus transformadas
originales = [UNO, DOS, TRES]
polares = [pnan1, pnan2, pnan3]
titulos = ["UNO", "DOS", "TRES"]

for i in range(3):
    # Imagen original con centro
    axes[i, 0].imshow(originales[i], cmap='inferno')
    axes[i, 0].scatter(*centro1[::-1], c='r', s=10)  # invertir para (y, x)
    axes[i, 0].set_title(f"Imagen original - {titulos[i]}")
    axes[i, 0].axis('off')

    # Imagen polar
    axes[i, 1].imshow(polares[i], cmap='inferno', aspect='auto')
    axes[i, 1].set_title(f"Coordenadas polares - {titulos[i]}")
    axes[i, 1].set_xlabel("r")
    axes[i, 1].set_ylabel("θ")

plt.tight_layout()
plt.show()

#%% PROMEDIO ANGULAR
valores2 = []
for i in range(pnan2.shape[1]):  # Recorremos todos los radios (filas)
    val = np.nanmean(pnan2[:, i])  # Promedio en theta para cada r
    valores2.append(val)

Rr = np.linspace(0, pnan2.shape[1] - 1, pnan2.shape[1])

plt.figure(figsize=(8, 5))
plt.plot(Rr, valores2)
plt.xlabel("r (pixeles)")
plt.ylabel("Promedio en θ")
plt.title("Perfil radial promedio - DOS")
plt.grid(True)
plt.show()

#%% HEAVYWEIGHT BABYY (PESOS PESADOS)

valores33 = []
pesos = []

for i in range(pnan3.shape[1]):
    datos_validos = ~np.isnan(pnan3[:, i])
    val = np.nanmean(pnan3[:, i])
    peso = np.sum(datos_validos)  # calculo promedio y el peso de cada radio
    valores33.append(val)
    pesos.append(peso)
    
    
Rr = np.linspace(1, pnan3.shape[1] , pnan3.shape[1])
valores33 = np.array(valores33)
pesos = np.array(pesos)

plt.figure(figsize=(8, 5))
sc = plt.scatter(Rr, valores33, c=pesos, cmap='viridis', s=30 + 70 * (pesos / np.max(pesos)))
plt.colorbar(sc, label="Cantidad de datos válidos (peso)")
plt.xlabel("r (pixeles)")
plt.ylabel("Promedio en θ")
plt.title("Perfil radial promedio ponderado")
plt.grid(True)
plt.show()
#%%
plt.figure(figsize=(8, 5))
plt.plot(Rr, valores33)
plt.xlabel("r (pixeles)")
plt.ylabel("Promedio en θ")
plt.title("Perfil radial promedio - TRES")
plt.grid(True)
plt.show()
#%%
# Graficar
plt.figure(figsize=(10, 4))
plt.plot(np.arange(400, len(valores33)), valores_recortados, label='Valores recortados')
plt.plot(picos_indices, valores33[picos_indices], 'ro', label='Picos detectados')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Picos modo 3')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%

valores_recortados = valores33[400:]
picos_relativos, _ = find_peaks(valores_recortados, distance = 40)

# Índices absolutos para todo valores33
picos_indices = picos_relativos + 400

picos_r = Rr[picos_indices] #ojo aca
picos_valores = valores33[picos_indices]
picos_pesos = pesos[picos_indices]


sigma = 1 / np.sqrt(picos_pesos)
sigma = np.where(picos_pesos == 0, 1e6, sigma)

Rr_rec = Rr[400:]


factor = 1
picos_valores_scaled = picos_valores * factor
sigma_scaled = sigma * factor

picos_valores_log = np.log(picos_valores_scaled)
sigma_log = sigma_scaled / picos_valores_scaled 
sigma = sigma_scaled

def modelo_lin(x, A, B):
    return A*x + B


popt, pcov = curve_fit(modelo_lin, picos_r, picos_valores_log, p0=[0.0001, 0.0], sigma=sigma, absolute_sigma=True)



y_fit = modelo_lin(Rr_rec, *popt)

A, B = popt

dA, dB = np.sqrt(np.diag(pcov))

plt.figure(figsize=(10, 6))
#plt.plot(Rr_rec, valores_recortados, label='Perfil radial')
plt.scatter(picos_r, picos_valores_log, color='darkmagenta', s=80, label='Picos detectados')

plt.plot(Rr_rec, y_fit, '--',
         label=f'Ajuste exponencial:\nk={A:.6f}±0.00038, B={B:.6f}' ,color='darkcyan')
plt.errorbar(picos_r, picos_valores_log, yerr=sigma, fmt='o', color='darkmagenta', label='Picos detectados ± sigma')

plt.xlabel('r (pixeles)')
plt.ylabel('Promedio en θ')
plt.title('Ajuste ponderado con curve_fit usando pesos')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()









#%%
valores22 = []
pesos22 = []

for i in range(pnan2.shape[1]):
    datos_validos = ~np.isnan(pnan3[:, i])
    val = np.nanmean(pnan2[:, i])
    peso = np.sum(datos_validos)  # calculo promedio y el peso de cada radio
    valores22.append(val)
    pesos22.append(peso)
    
    
Rr = np.linspace(1, pnan2.shape[1] , pnan2.shape[1])
valores22 = np.array(valores22)
pesos22 = np.array(pesos22)

plt.figure(figsize=(8, 5))
sc = plt.scatter(Rr, valores22, c=pesos, cmap='viridis', s=30 + 70 * (pesos / np.max(pesos)))
plt.colorbar(sc, label="Cantidad de datos válidos (peso)")
plt.xlabel("r (pixeles)")
plt.ylabel("Promedio en θ")
plt.title("Perfil radial promedio ponderado")
plt.grid(True)
plt.show()
#%%
plt.figure(figsize=(8, 5))
plt.plot(Rr, valores22)
plt.xlabel("r (pixeles)")
plt.ylabel("Promedio en θ")
plt.title("Perfil radial promedio - DOS")
plt.grid(True)
plt.show()

#%%

valores_recortados22 = valores22[400:]
picos_relativos22, _ = find_peaks(valores_recortados22, distance = 95)


#%%



# Índices absolutos para todo valores33
picos_indices22 = picos_relativos22 + 400

picos_r22 = Rr[picos_indices22] #ojo aca
picos_valores22 = valores22[picos_indices22]
picos_pesos22 = pesos22[picos_indices22]


sigma22 = 1 / np.sqrt(picos_pesos22)
sigma22 = np.where(picos_pesos22 == 0, 1e6, sigma22)

Rr_rec = Rr[400:]

#%%

plt.figure(figsize=(10, 6))
plt.plot(Rr_rec, valores_recortados22, label='Perfil radial (valores22)', color='black')
plt.scatter(picos_r22, picos_valores22, color='red', s=80, label='Picos detectados')

plt.xlabel('r (pixeles)')
plt.ylabel('Promedio en θ')
plt.title('Perfil radial y picos detectados')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
factor = 1
picos_valores_scaled22 = picos_valores22 * factor
sigma_scaled22 = sigma22 * factor

picos_valores_log22 = np.log(picos_valores_scaled22)
sigma_log22 = sigma_scaled22 / picos_valores_scaled22 
sigma22 = sigma_scaled22

def modelo_lin(x, A, B):
    return A*x + B


popt, pcov = curve_fit(modelo_lin, picos_r22, picos_valores_log22, p0=[0.0001, 0.0], sigma=sigma22, absolute_sigma=True)



y_fit22 = modelo_lin(Rr_rec, *popt)

A, B = popt

dA, dB = np.sqrt(np.diag(pcov))

plt.figure(figsize=(10, 6))
#plt.plot(Rr_rec, valores_recortados, label='Perfil radial')
plt.scatter(picos_r22, picos_valores_log22, color='g', s=80, label='Picos detectados')

plt.plot(Rr_rec, y_fit22, 'r--',
         label=f'Ajuste exponencial:\nA={A:.6f}, B={B:.6f}')
plt.errorbar(picos_r22, picos_valores_log22, yerr=sigma22, fmt='o', color='g', label='Picos detectados ± sigma')

plt.xlabel('r (pixeles)')
plt.ylabel('Promedio en θ')
plt.title('Ajuste ponderado con curve_fit usando pesos')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()















#%%
# ajuste de los picos del promedio pesado por el peso del radio 

def modelo_exp(r, A, k, C):
    return A * np.exp(-k * r) + C

valores_recortados = valores33[400:]
picos_relativos, _ = find_peaks(valores_recortados, distance = 40)

# Índices absolutos para todo valores33
picos_indices = picos_relativos + 400

picos_r = Rr[picos_indices]
picos_valores = valores33[picos_indices]
picos_pesos = pesos[picos_indices]

# Calcular sigma para curve_fit (incertidumbre)
sigma = 1 / np.sqrt(picos_pesos)
sigma = np.where(picos_pesos == 0, 1e6, sigma)

# Ajuste ponderado
popt, pcov = curve_fit(modelo_exp, picos_r, picos_valores,p0=[picos_valores.max(), 0.001, picos_valores.min()], sigma=sigma, absolute_sigma=True)

# Generar valores ajustados para graficar
r_fit = np.linspace(picos_r.min(), picos_r.max(), 100)
y_fit = modelo_exp(r_fit, *popt)

Rr_rec = Rr[400:]

A, k, C = popt

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(Rr_rec, valores_recortados, label='Perfil radial')
plt.scatter(picos_r, picos_valores, color='red', s=80, label='Picos detectados')

plt.plot(r_fit, y_fit, 'r--',
         label=f'Ajuste exponencial:\nA={A:.6f}, k={k:.6f}, C={C:.6f}')

plt.xlabel('r (pixeles)')
plt.ylabel('Promedio en θ')
plt.title('Ajuste ponderado con curve_fit usando pesos')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%











#%%
grado = 1
Rr_rec = Rr[400:]  # Rango de interés
valores_recortados = valores33[400:]
pesos_recortados = pesos[400:]

coef_fondo = np.polyfit(Rr_rec, valores_recortados, grado)
fondo_estimado = np.polyval(coef_fondo, Rr_rec)
valores_detrended = valores_recortados - fondo_estimado

# --- 2. Detectar picos máximos y mínimos ---
peaks_pos, _ = find_peaks(valores_detrended, distance=40)
peaks_neg, _ = find_peaks(-valores_detrended, distance=40)

r_picos_pos = Rr_rec[peaks_pos]
y_picos_pos = valores_detrended[peaks_pos]
peso_picos_pos = pesos_recortados[peaks_pos]

# Tiramos el primero y el último mínimo
peaks_neg = peaks_neg[1:-1]

r_picos_neg = Rr_rec[peaks_neg]
y_picos_neg = -valores_detrended[peaks_neg]  # invertir signo para ajustar en positivo
peso_picos_neg = pesos_recortados[peaks_neg]

# --- 3. Modelo exponencial ---
def modelo_exp(r, A, k, C):
    return A * np.exp(-k * r) + C

# --- 4. Calcular sigmas desde pesos ---
sigma_pos = 1 / np.sqrt(peso_picos_pos)
sigma_pos = np.where(peso_picos_pos == 0, 1e6, sigma_pos)

sigma_neg = 1 / np.sqrt(peso_picos_neg)
sigma_neg = np.where(peso_picos_neg == 0, 1e6, sigma_neg)

# --- 5. Ajustes exponenciales ---
popt_pos, _ = curve_fit(modelo_exp, r_picos_pos, y_picos_pos,
                        p0=[y_picos_pos.max(), 0.001, y_picos_pos.min()],
                        sigma=sigma_pos, absolute_sigma=True)

popt_neg, _ = curve_fit(modelo_exp, r_picos_neg, y_picos_neg,
                        p0=[y_picos_neg.max(), 1, y_picos_neg.min()],
                        sigma=sigma_neg, absolute_sigma=True)

# --- 6. Evaluar modelo ajustado ---
r_fit = Rr_rec
y_fit_pos = modelo_exp(r_fit, *popt_pos)
y_fit_neg = -modelo_exp(r_fit, *popt_neg)  # invertir de nuevo para el gráfico

# --- 7. Gráfico final ---
plt.figure(figsize=(10, 6))
plt.plot(Rr_rec, valores_detrended, color='black', label='Perfil radial (sin tendencia)')

# Picos positivos
plt.scatter(r_picos_pos, y_picos_pos, color='red', label='Máximos locales')
plt.plot(r_fit, y_fit_pos, 'r--', label=f'Ajuste máximos:\nA={popt_pos[0]:.6f}, k={popt_pos[1]:.6f}, C={popt_pos[2]:.6f}')

# Picos negativos
plt.scatter(r_picos_neg, -y_picos_neg, color='blue', label='Mínimos locales')
plt.plot(r_fit, y_fit_neg, 'b--', label=f'Ajuste mínimos:\nA={popt_neg[0]:.6f}, k={popt_neg[1]:.6f}, C={-popt_neg[2]:.6f}'
         if popt_neg[2] < 0 else
         f'Ajuste mínimos:\nA={popt_neg[0]:.6f}, k={popt_neg[1]:.6f}, C={popt_neg[2]:.6f}')

plt.xlabel('r (pixeles)')
plt.ylabel('Promedio en θ (sin tendencia)')
plt.title('Ajustes exponenciales a máximos y mínimos del perfil radial (r ≥ 400)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- 8. (Opcional) Ver tendencia eliminada ---
plt.figure(figsize=(8, 4))
plt.plot(Rr_rec, valores_recortados, label='Original')
plt.plot(Rr_rec, fondo_estimada := np.polyval(coef_fondo, Rr_rec), '--', label=f'Tendencia (grado {grado})')
plt.plot(Rr_rec, valores_detrended, label='Sin tendencia')
plt.xlabel("r (pixeles)")
plt.ylabel("Promedio en θ")
plt.title("Tendencia eliminada del perfil radial")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()










#%%
# Promedio en θ para cada r (pnan2)
valores2 = []
for i in range(pnan2.shape[1]):
    val = np.nanmean(pnan2[:, i])
    valores2.append(val)

# Promedio en θ para cada r (pnan3)
valores3 = []
for i in range(pnan3.shape[1]):
    val = np.nanmean(pnan3[:, i])
    valores3.append(val)

# Eje de radios (en píxeles)
Rr = np.linspace(0, pnan2.shape[1] - 1, pnan2.shape[1])

# Graficar ambos perfiles
plt.figure(figsize=(10, 6))
plt.plot(Rr, valores2, label='DOS', color='tomato')
plt.plot(Rr, valores3, label='TRES', color='dodgerblue')
plt.xlabel("r (pixeles)")
plt.ylabel("Promedio en θ")
plt.title("Perfiles radiales promedio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%

# perfiles en tita = 150
plt.figure(figsize=(10, 4))

theta = np.arange(pnan1.shape[1])  # eje horizontal: columnas

#plt.plot(theta, pnan1[150, :], label='UNO', color='tomato')
plt.plot(theta, pnan2[150, :], label='DOS', color='gold')
plt.plot(theta, pnan3[150, :], label='TRES', color='limegreen')

plt.title("Corte radial en tita = 150 px")
plt.xlabel("r (pixeles)")
plt.ylabel("Intensidad")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% exponenciales y picos
exp2 = pnan2[150, 400:]
exp3 = pnan3[150, 400:]

peaks2, _ = find_peaks(exp2)  
peaks3, _ = find_peaks(exp3)


#%% plot testeador
R = np.linspace(400, pnan3.shape[1], pnan3.shape[1]-400)
plt.plot(R, exp3)
plt.title("exp3 en θ = 150 y r > 400")
plt.xlabel("r")
plt.ylabel("Intensidad")
plt.grid(True)
plt.show()

#%% restar medias
prom2 = np.mean(exp2)
prom3 = np.mean(exp3)

exp20 = exp2 - prom2
exp30 = exp3 - prom3

plt.figure()
plt.plot(R, np.abs(exp30))
plt.title("exponencial menos media para modo 3")
plt.xlabel("radio")
plt.ylabel("Intensidad")
plt.grid(True)
plt.show()


#%% media restada

peaks20, _ = find_peaks(exp20) 
peaks30, _ = find_peaks(exp30)

radios = np.arange(400, 400 + len(exp3))

# Modelo exponencial
def modelo_exp(r, A, k, C):
    return A * np.exp(-k * r) + C

# Extraer datos en los picos
r_picos30 = radios[peaks30]
y_picos30 = exp30[peaks30]

# Ajuste exponencial
popt, pcov = curve_fit(modelo_exp, r_picos30, y_picos30, p0=[y_picos30.max(), 0.01, y_picos30.min()])

# Evaluar el modelo ajustado en un rango continuo
r_fit = np.linspace(r_picos30.min(), r_picos30.max(), 200)
y_fit = modelo_exp(r_fit, *popt)

# Gráfico
plt.figure(figsize=(10, 5))
plt.plot(radios, exp30, label='exp3 - media (θ = 150)')
plt.scatter(r_picos30, y_picos30, color='red', label='Picos detectados')
plt.plot(r_fit, y_fit, 'k--', label=f'Ajuste exponencial\nA={popt[0]:.6f}, k={popt[1]:.6f}, C={popt[2]:.6f}')
plt.xlabel("r (pixeles)")
plt.ylabel("Intensidad")
plt.title("Ajuste exponencial sobre los picos de exp3 - media")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()






#%% ajuste oara los maximos sin retsar media
peaks2, _ = find_peaks(exp2) 
peaks3, _ = find_peaks(exp3)

radios = np.arange(400, 400 + len(exp3))

def modelo_exp(r, A, k, C):
    return A * np.exp(-k * r) + C

r_picos3 = radios[peaks3]
y_picos3 = exp3[peaks3]

popt_max, pcov = curve_fit(modelo_exp, r_picos3, y_picos3, p0=[y_picos3.max(), 0.001, y_picos30.min()])

#r_fit = radios  #np.linspace(r_picos3.min(), r_picos3.max(), 200)
y_fit = modelo_exp(radios, *popt_max)

# Gráfico
plt.figure(figsize=(10, 5))
plt.plot(radios, exp3, label='exp3 (θ = 150)')
plt.scatter(r_picos3, y_picos3, color='red', label='Picos detectados')
plt.plot(radios, y_fit, 'k--', label=f'Ajuste exponencial\nA={popt[0]:.6f}, k={popt[1]:.6f}, C={popt[2]:.6f}')
plt.xlabel("r (pixeles)")
plt.ylabel("Intensidad")
plt.title("Ajuste exponencial sobre los picos de exp3")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%% ajuste para los minimos

negpeaks2, _ = find_peaks(-exp2) 
negpeaks3, _ = find_peaks(-exp3)

negpeaks20, _ = find_peaks(-exp20) 
negpeaks30, _ = find_peaks(-exp30)

radios = np.arange(400, 400 + len(exp3))

def modelo_exp(r, A, k, C):
    return A * np.exp(-k * r) + C

r_picosneg3 = radios[negpeaks3]
y_picosneg3 = -exp3[negpeaks3]

r_picosneg30 = radios[negpeaks30]
y_picosneg30 = -exp3[negpeaks30]


popt_min, pcov = curve_fit(modelo_exp, r_picosneg3, y_picosneg3, p0=[y_picosneg3.max(), 0.0001, y_picosneg30.min()])

#r_fit = radios  #np.linspace(r_picos3.min(), r_picos3.max(), 200)
y_fit = modelo_exp(radios, *popt)

# Gráfico
plt.figure(figsize=(10, 5))
plt.plot(radios, -exp3, label='exp3 (θ = 150)')
plt.scatter(r_picosneg3, y_picosneg3, color='red', label='Picos detectados')
plt.plot(radios, y_fit, 'k--', label=f'Ajuste exponencial\nA={popt[0]:.6f}, k={popt[1]:.6f}, C={popt[2]:.6f}')
plt.xlabel("r (pixeles)")
plt.ylabel("Intensidad")
plt.title("Ajuste exponencial sobre los picos de -exp3")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%% plotear todo junto 

# Ajuste sobre máximos
y_fit_max = modelo_exp(radios, *popt_max)

# Ajuste sobre mínimos (invertido porque se ajustó con -exp3)
y_fit_min = -modelo_exp(radios, *popt_min)

# Gráfico
plt.figure(figsize=(10, 5))
plt.plot(radios, exp3, label='exp3 (θ = 150)', color='black')

# Picos máximos
plt.scatter(r_picos3, y_picos3, color='red', label='maximos')
plt.plot(radios, y_fit_max, 'r--',
         label=f'ajuste por maximos:\nA={popt_max[0]:.6f}, k={popt_max[1]:.6f}, C={popt_max[2]:.6f}')

# Picos mínimos
plt.scatter(r_picosneg3, exp3[negpeaks3], color='blue', label='minimos')
plt.plot(radios, y_fit_min, 'b--',
         label=f'Ajuste por minimos:\nA={popt_min[0]:.6f}, k={popt_min[1]:.6f}, C={-popt_min[2]:.6}' if popt_min[2] < 0 else
               f'Ajuste por minimos:\nA={popt_min[0]:.6f}, k={popt_min[1]:.6f}, C={popt_min[2]:.6f}')

# Ejes y título
plt.xlabel("r (pixeles)")
plt.ylabel("Intensidad")
plt.title("Ajustes exponenciales sobre máximos y mínimos de exp3")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# 1. Eliminar tendencia polinómica
grado = 1  
coef_fondo = np.polyfit(radios, exp3, grado)
fondo_estimado = np.polyval(coef_fondo, radios)
exp3_detrended = exp3 - fondo_estimado

# 2. Detectar picos máximos y mínimos
peaks3, _ = find_peaks(exp3_detrended)
negpeaks3, _ = find_peaks(-exp3_detrended)

# 3. Datos para el ajuste
r_picos3 = radios[peaks3]
y_picos3 = exp3_detrended[peaks3]

r_picosneg3 = radios[negpeaks3]
y_picosneg3 = -exp3_detrended[negpeaks3]  # se ajusta en positivo

# 4. Modelo exponencial
def modelo_exp(r, A, k, C):
    return A * np.exp(-k * r) + C

# 5. Ajuste sobre máximos
popt_max, _ = curve_fit(modelo_exp, r_picos3, y_picos3,
                        p0=[y_picos3.max(), .01, y_picos3.min()])

# 6. Ajuste sobre mínimos
popt_min, _ = curve_fit(modelo_exp, r_picosneg3, y_picosneg3,
                        p0=[y_picosneg3.max(), .01, y_picosneg3.min()])

# 7. Evaluar ambos ajustes
y_fit_max = modelo_exp(radios, *popt_max)
y_fit_min = -modelo_exp(radios, *popt_min)  # revertimos el signo

# 8. Gráfico combinado
plt.figure(figsize=(10, 5))
plt.plot(radios, exp3_detrended, color='black', label='exp3 sin tendencia')

# Picos máximos
plt.scatter(r_picos3, y_picos3, color='red', label='Máximos locales')
plt.plot(radios, y_fit_max, 'r--',
         label=f'Ajuste máximos:\nA={popt_max[0]:.6f}, k={popt_max[1]:.6f}, C={popt_max[2]:.6f}')

# Picos mínimos
plt.scatter(r_picosneg3, -y_picosneg3, color='blue', label='Mínimos locales')
plt.plot(radios, y_fit_min, 'b--',
         label=f'Ajuste mínimos:\nA={popt_min[0]:.6f}, k={popt_min[1]:.6f}, C={-popt_min[2]:.6f}'
         if popt_min[2] < 0 else
         f'Ajuste mínimos:\nA={popt_min[0]:.6f}, k={popt_min[1]:.6f}, C={popt_min[2]:.6f}')

plt.title("Ajuste exponencial sobre máximos y mínimos de exp3 (sin tendencia)")
plt.xlabel("r (pixeles)")
plt.ylabel("Intensidad (sin tendencia)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 9. (Opcional) Ver la tendencia eliminada
plt.figure(figsize=(8, 4))
plt.plot(radios, exp3, label='exp3 original')
plt.plot(radios, fondo_estimado, '--', label=f'Tendencia polinómica (grado {grado})')
plt.plot(radios, exp3_detrended, label='exp3 sin tendencia')
plt.xlabel("r (pixeles)")
plt.ylabel("Intensidad")
plt.title("Tendencia polinómica eliminada")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
