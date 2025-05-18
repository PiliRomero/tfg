import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from funciones.fun import *


st.title("Señales")

st.write("Una **señal** es una magnitud que se puede representar como función de una o más variables independientes, por ejemplo, el tiempo o el espacio. Aquí se tratarán señales que varían únicamente en función del tiempo.")

st.header("Clasificación de las señales")


texto1=r"""
#### **Señales continuas y discretas**
*   Señales analógicas (continuas en el tiempo): están definidas en cada instante del tiempo.
* Señales discretas en el tiempo: sólo se definen para determinados instantes temporales.

Una señal continua o discreta en el tiempo puede tomar a su vez valores continuos  o discretos.
Las **señales digitales** son aquellas que son discretas en el tiempo y que además toman valores discretos.

#### **Señales determinísticas y aleatorias**
* Señales determinísticas: se conoce de antemano el valor de la señal para cada instante temporal.
* Señales aleatorias: el comportamiento de la señal no se puede describir mediante una fórmula matemática. La evolución de la señal a lo largo del tiempo tiene cierto grado de incertidumbre.

#### **Señales periódicas y no periódicas**
Para una señal periódica su valor se repite tras un desplazamiento temporal T, es decir $f(t)=f(t+T) \; \forall t$. El mínimo de T para el que se verifica la igualdad anterior es el **período** de la señal.

Un ejemplo de señal continua en el tiempo y periódica es la **oscilación armónica** que se describe por la ecuación:
$$
x_{a}(t) = Acos(Ωt+θ), \; t \: \epsilon \:\mathbb{R}
$$
donde:

*   A: amplitud de la sinusoide
*   Ω: frecuencia en $rad/s$
*   θ: fase en radianes

Si se considera la frecuencia (f) en ciclos por segundo ($Hz$) entonces la frecuencia $\Omega$ se puede escribir como $Ω=2 \pi f$) por lo que $x_{a}(t) = Acos(2\pi ft+θ)$. 

Utilizando la fórmula de Euler que establece la relación entre funciones trigonométricas y la exponencial compleja ($e^{ix}=cos(x) + i \;sen(x)$)

$$
x_{a}(t) = Acos(Ωt+θ) = \frac{A}{2}e^{j(Ωt+θ)} + \frac{A}{2}e^{-j(Ωt+θ)}, \; t \: \epsilon \:\mathbb{R}
$$

que es la suma de dos exponenciales complejas conjudadas (**fasores**) con la misma amplitud ($A/2$).

Una **Señal sinusoidal discreta en el tiempo** se expresa mediante la ecuación:

$$
x(n) = Acos(𝜔n+θ), \; n \: \epsilon \:\mathbb{Z}
$$

donde:

*   n: número de muestra
*   A: amplitud de la sinusoide
*   𝜔: frecuencia en $rad/muestra$
*   θ: fase en radianes
"""

st.markdown(texto1)

st.header("Conversión de señales analógicas a digitales")

texto2="""

1.   **Muestreo**: se convierte una señal continua en el tiempo ($x_{a}(t)$) en una señal discreta en el tiempo ($x(n)$) tomando muestras de la señal continua en unos instantes temporales determinados de modo que $x_{a}(nT)=x(n)$, donde T es el intervalo de muestreo.
2.   **Cuantificación**: se convierte una señal de valores continuos en una señal de valores discretos en instantes de tiempo discreto, es decir, se convierte en una señal digital. La diferencia entre las dos señales se denomina error de cuantificación. Las amplitudes permitidas en la señal discreta son los niveles de cuantificación y cuantos más niveles se usen mayor será la precisión y menor el error de cuantificación.
3.   **Codificación**: se representa cada valor de la señal digital por un código binario.

En los siguientes apartados se introducen las transformadas de Fourier y wavelet. La
transformada de Fourier muestra la relación entre la amplitud y la frecuencia de una señal, pero no indica en que instante temporal se dan estas componentes
de frecuencia. Esta información no es relevante para señales estacionarias, aquellas cuya frecuencia no cambia con el tiempo.

La transformada wavelet permite descomponer las señales, sean estacionarias o no, en componentes de tiempo y frecuencia. 
"""

st.markdown(texto2)
st.divider()

texto3=r"""
En esta página web se proporciona un conjunto de señales para probar las técnicas de preprocesado, agrupamiento y clasificación que se estudian.

Como parte del tratamiento, se armonizó el eje temporal de las señales objeto de estudio para que los datos de todas ellas coincidan en el tiempo. En primer lugar, como límite inferior se consideró el máximo de todos los instantes iniciales de las señales y como límite superior el mínimo de los instantes finales. 

Además, se llevó a cabo un muestreo para cada señal con una periodicidad de 0.01 milisegundos, interpolando los datos mediante Splines cúbicos.

Finalmente, también se llevó a cabo la normalización por la diferencia de los datos para, de este modo, obtener valores comprendidos en el intervalo [0,1].

$$
x'_{i} = \frac{x_{i}-x_{min}}{x_{max}-x_{min}}
$$
Los datos utilizados ya están convenientemente muestreados y normalizados. 
"""

st.markdown(texto3)

#datos=pd.read_csv("./datos/datos.csv")
#datosMuestreados=pd.read_csv("./datos/sMuestreadas.csv")
datosMuestreadosN=getDatosTF()
nombreSeries=getNombreSeriesTF()
tiposSeries=getTipoSeriesTF()

#datosMuestreadosN=pd.read_csv("./datos/sMuestreadasN.csv")
#nombreSeries=list(datosMuestreados.columns)[1:]
#tiposSeries=np.unique(list(map(lambda x: x[0:x.find('_')],nombreSeries)))

col1, col2 = st.columns(2)

with col1:
    optionTs=st.selectbox(
        "Seleccione un tipo de señal",
        options=tiposSeries, 
        index=1       
    )

    ls=[ns for ns in nombreSeries if optionTs in ns]

    optionS=st.selectbox(
        "Seleccione una señal",
        options=ls     
    )  

# Serie seleccionada
#serie=pd.Series(datos[optionS])
#serie.index=datos['t']
#serie.dropna(inplace=True)
#serieM=pd.Series(datosMuestreados[optionS])
#serieM.index=datosMuestreados['t']"""
serieMN=pd.Series(datosMuestreadosN[optionS])
serieMN.index=datosMuestreadosN['t']

with col2:
    fig1, ax1 = plt.subplots()
    ax1.plot(serieMN)
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Señal')
    ax1.set_title('Señal original '+ optionS)
    st.pyplot(fig1)

#with col2:
#    fig1, ax1 = plt.subplots()
#    ax1.plot(serie)
#    ax1.set_xlabel('Tiempo')
#    ax1.set_ylabel('Señal')
#    ax1.set_title('Señal original '+ optionS)
#    st.pyplot(fig1)

#fig2=plt.figure(figsize=(10,6))
#f1= fig2.add_subplot(221)
#f1.plot(serieM)
#f1.set_xlabel('Tiempo')
#f1.set_ylabel('Señal')
#f1.set_title('Señal muestreada: '+optionS)

#f2= fig2.add_subplot(222)
#f2.plot(serieMN)
#f2.set_xlabel('Tiempo')
#f2.set_ylabel('Señal')
#f2.set_title('Señal muestreada normalizada: '+optionS)

#st.pyplot(fig2)

st.subheader("Subir fichero de datos", divider="red")

texto4="""
Para probar los algoritmos propuestos puede subir su propio fichero .csv que debe cumplir las siguientes restricciones:
* Cada columna se corresponde con una señal.
* La trimera columna contendrá el tiempo.
* El nombre de las señales deberan tener el siguiente formato: TIPOSEÑAL_NONBRESEÑAL, es decir, para nombrar una señal en primer lugar
se pone el nombre de la tipología o clase de cada señal seguido por un guión bajo (_) y por el nombre de la señal. De esta forma es
posible etiquetar los ejemplos de entrenamiento para los algoritmos de aprendizaje automático supervisado.
* Se establece un mínimo de 20 observaciones para cada señal.
* El tamaño máximo del .csv debe ser menor de 200MB.
"""

with st.expander("Realizar mi propio análisis"):
    st.write(texto4)
    col1, col2 = st.columns(2)
    with col1:
        optionSeparador=st.selectbox(
            "Seleccione el caracter separador",
            options=[";",","], 
        )

    with col2:
        optionDecimal=st.selectbox(
            "Seleccione el separador decimal",
            options=[".",","]
        )

    uploaded_file = st.file_uploader("cargue un archivo csv",type='csv')
    if uploaded_file is not None:
        data=pd.read_csv(uploaded_file, sep=optionSeparador, decimal=optionDecimal)
        st.write(data)

        frecuencia=st.checkbox("Fijar frecuencia de muestreo")

        frec=None
        
        if frecuencia:
            frec=st.number_input("Insterte el valor de la frecuencia", value=None,format="%0.1f",placeholder="Inserte un número")

        data=getDatosExternosN(data,frec)
        st.write("Datos muestreados y morzalizados")
        st.write(data)
        if data is not None:
            setDatosExternos(data)
