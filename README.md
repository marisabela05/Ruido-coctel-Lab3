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
> + SNR para audio Isabela: 15.15 dB
> + SNR para audio Ana: 22.30 dB
> + SNR para audio Luna: 14.67 dB
Esto indica que el micrófono dos, indica mejor calidad de audio ya que la señal es significativamente mas fuerte que el ruido
