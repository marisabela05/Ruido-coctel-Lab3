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
Para cada microfono se muestra la representación en el dominio del tiempo de la señal de audio. En esta gráfica se pueden observar variaciones en la amplitud de la señal a lo largo del tiempo, lo que permite identificar momentos de mayor o menor intensidad en la grabación y la segunda gráfica muestra el análisis espectral de la señal mediante la Transformada Rápida de Fourier (FFT).
### Micrófono 1
#### *Gráfica temporal*
Audio de Ana 
![Imagen de WhatsApp 2025-03-05 a las 23 05 23_ce73418f](https://github.com/user-attachments/assets/d33c120d-6216-4e23-82f5-e22f56b0e894)
+ La señal muestra variaciones en la amplitud a lo largo del tiempo, con momentos de mayor intensidad alrededor de los 25-35 segundos.
+ Hay una distribución simétrica de la amplitud en torno a cero.
#### *Gráfica espectral*
![Imagen de WhatsApp 2025-03-06 a las 23 03 24_028314ce](https://github.com/user-attachments/assets/93b97a17-4542-41d4-921b-14650e968a03)
+ Tiene una fuerte presencia de frecuencias bajas y medias
+ Se pueden notar fluctuaciones en todo el espectro, lo que sugiere la presencia de ruido en la grabación.
+ A partir de aproximadamente 5 kHz, la magnitud del espectro comienza a decaer de manera gradual.
#### *Gráfica temporal*
Audio Luna
![Imagen de WhatsApp 2025-03-05 a las 23 13 55_ce94d599](https://github.com/user-attachments/assets/ed3601fd-58a6-4526-b8a0-f56ed021c069)
+ Al inicio, la amplitud es más estable y con menor variabilidad, mientras que hacia el final se observa mayor dispersión, lo que podría estar asociado a una variación en la intensidad del sonido o la aparición de ruido en la grabación.
#### *Gráfica espectral*
![Imagen de WhatsApp 2025-03-06 a las 23 06 21_22022cfa](https://github.com/user-attachments/assets/f619e980-a1d6-4b5f-8b86-9cb428432e4a)
+ A partir de los 5000 Hz, la magnitud disminuye progresivamente, pero se pueden notar pequeñas contribuciones en frecuencias más altas, lo que podría ser ruido o armónicos de la señal principal.
+ La estructura espectral es consistente con una señal de audio con predominancia en frecuencias bajas y medias.

#### Micrófono 3 
#### *Gráfica temporal*
![Imagen de WhatsApp 2025-03-05 a las 22 42 49_dc1c2fb8](https://github.com/user-attachments/assets/5fd0166e-fd97-4018-a1d6-13cdad36a22c)
+ Existen regiones con menor amplitud, que podrían corresponder a pausas o silencios.
+ Se puede observar un aumento en la energía de la señal alrededor de los 25-35 segundos, lo que podría indicar un incremento en la intensidad sonora en ese período.
#### *Gráfica espectral*
![Imagen de WhatsApp 2025-03-06 a las 23 01 12_1c27e7a8](https://github.com/user-attachments/assets/bf8b914b-c008-41b7-8873-bb427537d87b)
+ Aquí se representa la magnitud de las frecuencias en el eje vertical y la frecuencia en Hz en el eje horizontal.
+ La magnitud de las altas frecuencias disminuye progresivamente, lo que es típico en señales de voz, donde las bajas y medias frecuencias tienen mayor relevancia.
>Se observa que la mayor parte de la energía está concentrada en frecuencias bajas, aproximadamente por debajo de 5000 Hz.
>
