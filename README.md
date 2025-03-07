# Problema tipo fiesta coctel
## Descripción
Este laboratorio aborda la temática conocida como el problema de la "fiesta de cóctel" tiene como objetivo analizar la relación entre el ruido y tres señales de audio en un ambiente lleno de sonidos, como la mezcla de voces de tres personas hablando al mismo tiempo. Así mismo, se aborda el desafío de aislar una voz de interés sobre las tres señales. Para ello, se calculará la relación señal/ruido SNR y se aplicarán técnicas de análisis espectral y separación de fuentes, como el análisis de aomponentes independientes ICA y el beamforming, utilizando micrófonos distribuidos estratégicamente. En esta práctica comprenderemos los principios del procesamiento de señales en entornos ruidosos, evaluando el impacto del ruido en la calidad del audio y por último aplicar herramientas computacionales para la extracción de señales relevantes.

## Tener en cuenta
1. Es importante tomar medidas de localización y posición tanto de los participantes como de los micrófonos en la sala.
3. Se recomienda seleccionar un espacio con el menor ruido ambiental posible para maximizar el SNR y facilitar el procesamiento de las señales.
4. Es necesario grabar el "silencio" que para el cálculo del SNR será como el ruido de la señal.
5. Se debe verificar que los micrófonos estén grabando a la misma frecuencia, para este laboratorio se grabó a 44.1kHz.
6. Se utilizan las siguientes librerias:
  + Wfdb.
  + Numpy.
  + Pandas.
  + Matplotlib.
  + Librosa //Se usa para el procesamiento de audio.
  + Scipy //Importa correctamente la FFT.
  + Scikit-learn //Ayuda en el analisis de componentes independientes.
7. Se utiliza Jupyter NoteBook para dividir el código en partes y trabajar en ellas sin importar el orden: escribir, probar funciones, cargar un archivo en la memoria y procesar el contenido. Con lenguaje de Python.

## Procedimiento
### Configuración del sistema y captura de la señal
Inicialmente, se colocan tres micrófonos en la sala y se miden las distancias entre ellos, así como entre cada micrófono y los tres participantes, tal como se muestra en la imagen: 
![Imagen de WhatsApp 2025-03-05 a las 13 25 39_11386d4c](https://github.com/user-attachments/assets/117ef3e0-cc02-482f-85ee-7a78efc65599)

Posteriormente, estas mediciones serán útiles para realizar el análisis con beamforming.
Una vez registradas todas las distancias, se calibran los micrófonos a una frecuencia de muestreo de 44.1 kHz. Ahora se procede a grabar, se realizan dos grabaciones en cada micrófono, una de 10 segundos de "silencio" que será usada como el ruido, y otra que es la propia del ruido tipo cóctel, donde las tres participantes hablaron simultáneamente.

Al tener los archivos mp3 se procede a utilizar la funcion de *librosa* para importa los ruidos y audios de las señales.  
```python
#Un arreglo que contiene la señal de audio en forma de valores de amplitud, La tasa de muestreo del audio
señal_isa, sr_señal_isa = librosa.load("Audio-Isa.mp3", sr=None)  # Señal principal Isa,.
ruido_isa, sr_ruido_isa = librosa.load("Ruido-Isabela.mp3", sr=None)  # Ruido Isa
```
Lo mismo se hace para los 3 integrantes para importar y tener la señal en JupyterNotebook.

### Calculo de SNR 

Con al siguiente funcion se calcula el SNR para cada integrante, donde la funcion recibe 2 argumentos y calcula la potencia tanto de la señal como el ruido y procede a retornar el valor del SNR.
```python
#Función para calcular el SNR, retorna ese valor
def calculate_snr(señal, ruido):
    
    if len(señal) != len(ruido): #Si la señal y el ruido tienen longitudes diferentes, se recorta la más larga para que ambas tengan el mismo número de muestras.
        min_length = min(len(señal), len(ruido))
        señal, ruido = señal[:min_length], ruido[:min_length]
    
    pot_señal = np.mean(señal**2)
    pot_ruido = np.mean(ruido**2)
    
    snr = 10 * np.log10(pot_señal / pot_ruido)#calculo SNR
    return snr
# SNR para Isabela
    snr_isa = calculate_snr(señal_isa, ruido_isa)
    print(f"SNR para audio Isabela: {snr_isa:.2f} dB")

# SNR para Ana
    snr_ana = calculate_snr(señal_ana, ruido_ana)
    print(f"SNR para audio Ana: {snr_ana:.2f} dB")

# SNR para Luna
    snr_luna = calculate_snr(señal_luna, ruido_luna)
    print(f"SNR para audio Luna: {snr_luna:.2f} dB")
```
> + SNR para microfono 3 audio Isabela: 15.15 dB
> + SNR para microfono 1 Ana: 22.30 dB
> + SNR para microfono 2 audio Luna: 14.67 dB
>
Esto indica que el *micrófono dos*, tiene mejor calidad de audio ya que la señal es significativamente más fuerte que el ruido, hay dos persona más cerca del *micrófono dos* que a los otros dos micrófonos. Aunque la distancia es un factor importante en la calidad del audio, el SNR también está influenciado por otros elementos como el ruido ambiental, la dirección de la voz y la acústica del lugar. Sin embargo a menor distancia, mayor SNR y mejor calidad del audio.
### Procesamiento de la señal 
Para cada microfono se muestra la representación en el dominio del tiempo de la señal de audio. En esta gráfica se pueden observar variaciones en la amplitud de la señal a lo largo del tiempo, lo que permite identificar momentos de mayor o menor intensidad en la grabación y la segunda gráfica muestra el análisis espectral de la señal mediante la Transformada Rápida de Fourier (FFT). El codigó que nos facilita ver la grafica temporal y espectral es el siguiente: 
```python
# Cargar la señal de audio
audio = "Audio-Isa.mp3"  # Cambia según el archivo a analizar
señal, sr = librosa.load(audio, sr=None) 

# === 1. ANÁLISIS TEMPORAL ===
#Aquí se visualiza la forma de onda del audio en el dominio del tiempo
plt.figure(figsize=(12, 4))
librosa.display.waveshow(señal, sr=sr, alpha=0.75)
plt.title("Forma de onda - Análisis temporal")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show()

# === 2. ANÁLISIS ESPECTRAL ===
#Aquí transformamos la señal del dominio del tiempo al dominio de la frecuencia usando la Transformada Rápida de Fourier:
# Aplicar FFT y calcular la magnitud del espectro
N = len(señal)
frec = np.fft.rfftfreq(N, d=1/sr)  # Frecuencias asociadas
fft = np.fft.rfft(señal)  # Aplicar FFT
magnitud = np.abs(fft)  # Obtener magnitud

# Graficar el espectro
#Aquí visualizamos el contenido espectral de la señal
plt.figure(figsize=(12, 4))
magnitud_db = 20 * np.log10(magnitud + 1e-6)  # Se suma 1e-6 para evitar log(0)
plt.plot(frec, magnitud_db, color='red', alpha=0.75)
plt.title("Análisis espectral - FFT")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud db")
plt.xlim(0, sr/2)  # Mostrar hasta la frecuencia de Nyquist
plt.grid()
plt.show()
```
### Micrófono 1
#### *Gráfica temporal* 
![Imagen de WhatsApp 2025-03-05 a las 23 05 23_ce73418f](https://github.com/user-attachments/assets/d33c120d-6216-4e23-82f5-e22f56b0e894)
+ La señal muestra variaciones en la amplitud a lo largo del tiempo, con momentos de mayor intensidad alrededor de los 25-35 segundos.
+ Hay una distribución simétrica de la amplitud en torno a cero.
#### *Gráfica espectral*
![Imagen de WhatsApp 2025-03-06 a las 23 03 24_028314ce](https://github.com/user-attachments/assets/93b97a17-4542-41d4-921b-14650e968a03)
+ Tiene una fuerte presencia de frecuencias bajas y medias
+ Se pueden notar fluctuaciones en todo el espectro, lo que sugiere la presencia de ruido en la grabación.
+ A partir de aproximadamente 5 kHz, la magnitud del espectro comienza a decaer de manera gradual.
### Micrófono 2
#### *Gráfica temporal*
![Imagen de WhatsApp 2025-03-05 a las 23 13 55_ce94d599](https://github.com/user-attachments/assets/ed3601fd-58a6-4526-b8a0-f56ed021c069)
+ Al inicio, la amplitud es más estable y con menor variabilidad, mientras que hacia el final se observa mayor dispersión, lo que podría estar asociado a una variación en la intensidad del sonido o la aparición de ruido en la grabación.
#### *Gráfica espectral*
![Imagen de WhatsApp 2025-03-06 a las 23 06 21_22022cfa](https://github.com/user-attachments/assets/f619e980-a1d6-4b5f-8b86-9cb428432e4a)
+ El rango de magnitud va aproximadamente de -80 dB a 60 dB, lo que indica que se están considerando componentes débiles y fuertes en la señal.
+ Es posible que este espectro pertenezca a un micrófono con mejor respuesta en agudos o a una señal con menor contenido de ruido en frecuencias bajas.

#### Micrófono 3 
#### *Gráfica temporal*
![Imagen de WhatsApp 2025-03-05 a las 22 42 49_dc1c2fb8](https://github.com/user-attachments/assets/5fd0166e-fd97-4018-a1d6-13cdad36a22c)
+ Existen regiones con menor amplitud, que podrían corresponder a pausas o silencios.
+ Se puede observar un aumento en la energía de la señal alrededor de los 25-35 segundos, lo que podría indicar un incremento en la intensidad sonora en ese período.
#### *Gráfica espectral*
![Imagen de WhatsApp 2025-03-06 a las 23 01 12_1c27e7a8](https://github.com/user-attachments/assets/bf8b914b-c008-41b7-8873-bb427537d87b)
+ Aquí se representa la magnitud db de las frecuencias en el eje vertical y la frecuencia en Hz en el eje horizontal.
+ Tiene menor caída en frecuencias altas y menos ruido en bajas frecuencias.
+ La pendiente de atenuación es más gradual, lo que indica que se mantienen componentes de alta frecuencia sin una caída abrupta entre frecuencias de 500 Hz - 15 kHz

### Separación de fuentes
Para este apartado se implementa un algoritmo de separación de fuentes de audio utilizando técnicas de GCC-PHAT, beamforming y Análisis de Componentes Independientes (ICA).
#### Retardos entre micrófonos con GCC-PHAT
```python
# Función para calcular el retardo con GCC-PHAT
def gcc_phat(sig1, sig2, fs):
    n = len(sig1) + len(sig2)
    
    SIG1 = fft(sig1, n=n)
    SIG2 = fft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R /= np.abs(R)
    cc = np.real(ifft(R))
    max_shift = len(sig1) // 2
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift]))
    delay = np.argmax(cc) - max_shift
    return delay

# Calcular los retardos entre micrófonos
delay_12 = gcc_phat(mic1, mic2, sr1)
delay_13 = gcc_phat(mic1, mic3, sr1)
delays = [0, delay_12, delay_13]
```
>Esta función calcula el tiempo de retardo entre dos micrófonos usando la técnica GCC-PHAT (Generalized Cross-Correlation with Phase Transform).
>Convierte las señales al dominio de la frecuencia (fft), calcula la correlación cruzada y obtiene el desplazamiento temporal entre las señales.
>Calcula los retardos entre los micrófonos y almacena los valores en delays.
>
#### Aplicación Beamforming 
```python
# Aplicar Beamforming (Delay-and-Sum)
def delay_and_sum(mics, delays):
    aligned_signals = [np.roll(m, d) for m, d in zip(mics, delays)]
    return np.mean(aligned_signals, axis=0)

beamformed_signal = delay_and_sum([mic1, mic2, mic3], delays)
```
>Alinea las señales de los micrófonos corrigiendo los retardos (np.roll).
>Promedia las señales alineadas para reforzar la fuente de interés y reducir el ruido.
>
#### ICA
```python
# Crear matriz de observaciones para ICA
X = np.vstack([mic1, mic2, mic3]).T

# Aplicar ICA para separar fuentes
ica = FastICA(n_components=3)
S_ica = ica.fit_transform(X)
sf.write("voz_separada_beamforming.wav", beamformed_signal, sr1)

# Guardar la señal más limpia
sf.write("voz_extraida.wav", S_ica[:, 0], sr1)
```
>Aplica Análisis de Componentes Independientes (ICA) para separar las señales en diferentes fuentes independientes (por ejemplo, voces separadas).
>Y las guarda con la funcion SF.WRITE.
>
### Calculo SNR final
#### Calculo de retardos entre miccrofonos y aplicación de beamforming 

```python
def calcular_retraso(distancias, velocidad, sr):
    return tuple(int(d / velocidad * sr) for d in distancias)

distancias = [0, 1.72,3.46]  # Distancia entre micrófonos en metros
velocidad_sonido = 343  # Velocidad del sonido en m/s
retraso = calcular_retraso(distancias, velocidad_sonido, sr1)
def beamforming(signals, delay):
    num_mics = signals.shape[1]
    beamformed_signal = np.zeros(len(signals))
    for i, delay_i in enumerate(delay):
        beamformed_signal += np.roll(signals[:, i], delay_i)
    return beamformed_signal / num_mics

# Asegurar que las tres señales tengan la misma longitud
longitud_max = max(len(señal_isa), len(señal_ana), len(señal_luna))
audio1 = np.pad(señal_isa, (0, longitud_max - len(señal_isa)))
audio2 = np.pad(señal_ana, (0, longitud_max - len(señal_ana)))
audio3 = np.pad(señal_luna, (0, longitud_max - len(señal_luna)))

# Crear la matriz de señales: cada columna representa un micrófono
audio_mix = np.vstack((audio1,audio2, audio3)).T

# Aplicar beamforming
beamformed_signal = beamforming(audio_mix, retraso)
sf.write("voz_separada_beamforming.wav", beamformed_signal, sr1)
```
>El resultado (retraso) es un número entero que indica cuántas muestras debe desplazarse cada señal.
>Se aplica beamforming a la señal combinada y se guarda el audio procesado (voz_separada_beamforming.wav).
#### Aplicación de ICA
```python
# Aplicar Análisis de Componentes Independientes (ICA)
ica = FastICA(n_components=3)
señales_separadas = ica.fit_transform(audio_mix)
señal_ica = señales_separadas[:, 0]  # Tomamos la primera señal separada
sf.write("voz_extraida.wav", señal_ica, sr1)
```
>FastICA se usa para separar las fuentes de audio de la mezcla de micrófonos.
>Se extrae la primera señal independiente (señal_ica), que corresponde a la fuente de interés.
>Se guarda la señal extraída en voz_extraida.wav.
#### Suma de ruidos y calculos de SNR FINAL
```python
# Asegurar que ambas señales de ruido tengan la misma longitud
longitud_max_ruido = max(len(ruido_isa), len(ruido_ana), len(ruido_luna))
ruido1 = np.pad(ruido_isa, (0, longitud_max_ruido - len(ruido_isa)))
ruido2 = np.pad(ruido_ana, (0, longitud_max_ruido - len(ruido_ana)))
ruido3 = np.pad(ruido_luna, (0, longitud_max_ruido - len(ruido_luna)))

# Sumar los ruidos
señal_suma = ruido1 + ruido2 + ruido3

# Calcular SNR final
SNR_FINAL_BEAM = calculate_snr(beamformed_signal, señal_suma)
SNR_FINAL_ICA = calculate_snr(señal_ica, señal_suma)

print(Fore.BLUE + f"SNR FINAL después de Beamforming: {SNR_FINAL_BEAM} dB")
print(Fore.BLUE + f"SNR FINAL después de ICA: {SNR_FINAL_ICA} dB")
```
>Suma los ruidos captados en cada micrófono y calcula SNR después de aplicar beamforming e ICA para medir la mejora en la calidad de la señal.
>1. SNR FINAL después de Beamforming: 8.783320207179363 dB
>2. SNR FINAL después de ICA: 40.53800582885742 dB
