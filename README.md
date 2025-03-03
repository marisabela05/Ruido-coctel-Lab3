# Problema tipo fiesta coctel
## Descripción
Este laboratorio aborda la temática conocida como el problema de la "fiesta de cóctel" tiene como objetivo analizar la relación entre el ruido y tres señales de audio en un ambiente lleno de sonidos, como la mezcla de voces de tres personas hablando al mismo tiempo. Así mismo, se aborda el desafío de aislar una voz de interés sobre las tres señales. Para ello, se calculará la relación señal/ruido SNR y se aplicarán técnicas de análisis espectral y separación de fuentes, como el análisis de aomponentes independientes ICA y el beamforming, utilizando micrófonos distribuidos estratégicamente. En esta práctica comprenderemos los principios del procesamiento de señales en entornos ruidosos, evaluando el impacto del ruido en la calidad del audio y por último aplicar herramientas computacionales para la extracción de señales relevantes.

## Tener en cuenta
1. Es importante tomar medidas de localización y posición tanto de los participantes como de los micrófonos en la sala.
3. Se recomienda seleccionar un espacio con el menor ruido ambiental posible para maximizar el SNR y facilitar el procesamiento de las señales.
4. Es necesario grabar el "silencio" que para el cálculo del SNR será como el ruido de la señal.
5. Se debe verificar que los micrófonos estén grabando a la misma frecuencia, para este laboratorio se grabó a 44.1kHz.

## Procedimiento
### Localización y toma de las señales
Inicialmente, se colocaron tres micrófonos en la sala y se midieron las distancias entre ellos, así como entre cada micrófono y los tres participantes, tal como se muestra en la figura *****. Posteriormente, estas mediciones serán útiles para realizar el análisis con beamforming.

**********************colocar fotos

Una vez registradas todas las distancias, se calibraron los micrófonos a una frecuencia de muestreo de 44.1 kHz. Ahora síse procede a grabar, se realizaron dos grabaciones, una de 10 segundos de "silencio" que será usada como el ruido, y otra que es la propia del ruido tipo cóctel, donde los tres participantes hablaron simultáneamente.

### Análisis de las señales
Luego de tener las grabaciones 

