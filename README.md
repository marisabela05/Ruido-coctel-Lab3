# Problema tipo fiesta coctel
## 1.Descripción
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

## 2. Procedimiento
### 2.1 Configuración del sistema y captura de la señal
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

### 2.2 Calculo de SNR audio original

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
### 2.3 Procesamiento de la señal 
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
#### Micrófono 1
![image](https://github.com/user-attachments/assets/42e02e97-7996-42cd-8014-bdb359067788)

**Gráfica temporal** 
+ La señal muestra variaciones en la amplitud a lo largo del tiempo, con momentos de mayor intensidad alrededor de los 25-35 segundos.
+ Hay una distribución simétrica de la amplitud en torno a cero.

**Gráfica espectral**
+ Tiene una fuerte presencia de frecuencias bajas y medias
+ Se pueden notar fluctuaciones en todo el espectro, lo que sugiere la presencia de ruido en la grabación.
+ A partir de aproximadamente 5 kHz, la magnitud del espectro comienza a decaer de manera gradual.
#### Micrófono 2
![image](https://github.com/user-attachments/assets/ffea4c21-03d0-4c45-96a2-10092712d003)

**Gráfica temporal**
+ Al inicio, la amplitud es más estable y con menor variabilidad, mientras que hacia el final se observa mayor dispersión, lo que podría estar asociado a una variación en la intensidad del sonido o la aparición de ruido en la grabación.

**Gráfica espectral**
+ El rango de magnitud va aproximadamente de -80 dB a 60 dB, lo que indica que se están considerando componentes débiles y fuertes en la señal.
+ Es posible que este espectro pertenezca a un micrófono con mejor respuesta en agudos o a una señal con menor contenido de ruido en frecuencias bajas.

#### Micrófono 3 
![image](https://github.com/user-attachments/assets/8e7a6ea6-ed97-40bc-b20e-f1f585b49b7d)

**Gráfica temporal**
+ Existen regiones con menor amplitud, que podrían corresponder a pausas o silencios.
+ Se puede observar un aumento en la energía de la señal alrededor de los 25-35 segundos, lo que podría indicar un incremento en la intensidad sonora en ese período.

**Gráfica espectral**
+ Aquí se representa la magnitud db de las frecuencias en el eje vertical y la frecuencia en Hz en el eje horizontal.
+ Tiene menor caída en frecuencias altas y menos ruido en bajas frecuencias.
+ La pendiente de atenuación es más gradual, lo que indica que se mantienen componentes de alta frecuencia sin una caída abrupta entre frecuencias de 500 Hz - 15 kHz

La combinación de ambos análisis nos permiten tener una interpretación más completa de la señal: El análisis temporal ayuda a detectar cuándo ocurre un evento en la señal cómo varía la amplitud de la señal con el tiempo. En cambio el análisis espectral ayuda a entender qué componentes frecuenciales están,cómo se distribuyen las diferentes frecuencias dentro de esta

### 2.4 Separación de fuentes
Se debe tener en cuenta en este ítem que primero esto se aplica para cada microfono y segundo, que el beamforming es una técnica que se va a utilizar en el procesamiento de señal con arreglos de micrófonos para dirigir la captación de sonido en una dirección específica el cual se basa en la manipulación de la fase y amplitud de las señales recibidas por cada micrófono para mejorar la captación de una fuente de sonido en particular y atenuar el ruido o interferencias provenientes de otras direcciones.
```python
# Distancias entre micrófonos
distancia = {"Isa": [0, 3.46, 3.50], "Ana": [1.72, 0, 3.50], "Luna": [3.46, 1.72, 0]}  # en metros
v_sonido = 343  # m/s

# Procesar cada señal por separado
for name, señal in zip(["Isa", "Ana", "Luna"], [señal_isa, señal_ana, señal_luna]):
    # Beamforming con alineación por retardo
    delays = np.array(distancia[name]) / v_sonido  # Convertir distancia a tiempo de retardo
    t_muestras = [int(sr_señal_isa * d) for d in delays]  # Convertir tiempo a muestras
    
    # Aplicar beamforming con Delay-and-Sum
    s_alineada = np.zeros_like(señal)
    for shift in t_muestras:
        s_alineada += np.roll(señal, shift) / len(t_muestras)  # Promedio de señales alineadas
    
    write(f'beamformed_{name}.wav', sr_señal_isa, (s_alineada * 32767).astype(np.int16))
```
Tambien, se realiza la aplicacion del Análisis de Componentes Independientes (ICA) el cual al ser un método de separación ciega de fuentes nos ayuda a descomponer una mezcla de señales en sus componentes originales asumiendo que son estadísticamente independientes. ICA nos fue útil para cuando se tienen múltiples señales que han sido captadas por diferentes micrófonos y se desea recuperar cada fuente original sin conocer sus características exactas.
```python
    ica = FastICA(n_components=1)
    s_separada = ica.fit_transform(señal.reshape(-1, 1)).flatten()
    write(f'ica_{name}.wav', sr_señal_isa, (s_separada * 32767).astype(np.int16))
```

La representacion gráfica para cada microfono se ve en la siguiente imagen.
![image](https://github.com/user-attachments/assets/a7c419c7-9a70-4ebd-af43-f737137d7226)

### 2.4 Calculo SNR 
Después de aplicar las tecnicas de separacion queriamos evealuar los resultados comparando la señal aislada con la señal original utilizado la relación señal/ruido para cuantificar el desempeño de la separación
```python
   # Calcular SNR
    snr_beamforming = calculate_snr(s_alineada, señal)
    snr_ica = calculate_snr(s_separada, señal)
    print(f"SNR para {name} - Beamforming: {snr_beamforming:.2f} dB")
    print(f"SNR para {name} - ICA: {snr_ica:.2f} dB")
```
|   Microfono |SNR BEAMFORMING   | SNR ICA | SNR ORIGINAL |
|--------------|-----------|-----------|----------- |
| ISA (3)| -2.55dB  | 29.40dB |  15.15dB  |
| ANA (1)| -4.13dB  | 25.02dB | 22.30dB |
| LUNA (2)| -4.36dB  | 28.50dB |14.67dB|

#### Análisis
La técnica de **Beamforming** fue la menos efectiva para la separación del audio de los microfonos ya que en todos los casos disminuyó con respecto a la original, por tanto podria ser que hubo un mal ajuste en los microfonos y que la señal quedo por debajo del nivel de ruido. Es probable que el algoritmo de beamforming no esté bien configurado (mala selección de retardos o pesos) o que las señales provengan de múltiples fuentes en direcciones difíciles de filtrar.

En cuanto la tecnica de **ICA** se evidencia que pudo separar mejor las fuentes de audio, eliminando ruido y preservando la señal útil. La única excepción que se tuvo fue en el microfono de ANA (1), donde la SNR final es menor que la original, lo que podría deberse a que el método no pudo separar correctamente la señal deseada o que introdujo artefactos. El éxito en dos de tres micrófonos indica que es un buen método para la separación de fuentes.
