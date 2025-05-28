import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import math
from PaginaWeb.funciones.fun import *

st.title("Wavelets")


texto1=r"""
La transformada Wavelet permite analizar de modo eficiente señales no estacionarias y que presenten fenómenos transitorios y con alta frecuencia. Descomponen la señal (estacionaria o no estacionaria) en componentes de tiempo y frecuencia.

Mientras que la transformada de Fourier descompone la señal en componentes sinusoidales de distintas frecuencias, la transformada Wavelet trocea la señal en versiones reescaladas y trasladadas de una señal que se denominará Wavelet madre.

Una Wavelet es una señal oscilatoria de corta duración con energía finita ($\int_{-∞}^{+∞} \left\| ψ(t)\right\|^{2}dt \; <\; ∞$) y concentrada en un intervalo temporal.

En función de la señal que se va a estudiar se elige una Wavelet a la que se denominará **Wavelet madre** $\psi(t)$ y a partir de ella se pueden definir nuevas Wavelets añadiendo parámetros de escala (a)  y traslación (b):
$$
\psi_{a,b}(t)=\frac{1}{\sqrt{a}}\psi \left ( \frac{t-b}{a} \right), \; t \; \epsilon \;  \mathbb{R}, \; a \; b, \; \epsilon  \; \mathbb{R}, \; a>0  
$$
Las Wavelets $\psi_{a,b}(t)$ tienen la fisma forma que la wavelet madre pero diferente escala y distinta ubicación.  
"""
st.markdown(texto1)

st.header("Transformada de wavelet continua (CWT)")

texto2=r"""
Se define la transformada Wavelet continua mediante la ecuación: 
$$
CWT(a,b)=\int f(t) \psi_{a,b}^{*}(t)dt, \; con \; a, \; b \;  \epsilon  \; \mathbb{R}, \; a>0  
$$
donde $f(t)$ es una señal continua que se quiere analizar y * representa el complejo conjugado (misma componente real y la parte imaginaria cambiada de signo).

Cuando el factor de escala es pequeño ($a \;\epsilon \;(0,1))$ se tiene una Wavelet contraída y se captan los detalles de la serie que cambian rápidamente.  Los valores pequeños de $a$ se corresponden a altas frecuencias. Para valores grandes de $a$ (mayores que uno) las Wavelets son dilatadas que captan los detalles que cambian más lentamente (bajas frecuencias).

Hay que tener en cuenta que la transformada Wavelet no representa del todo la señal. Es necesario considerar la transformación 
de la función escala. Las funciones escala juegan el papel de funciones promedio.
$$
\phi_{a,b}(t)=\frac{1}{\sqrt{a}}\phi \left ( \frac{t-b}{a} \right), \; t \; \epsilon \;  \mathbb{R}, \; a \; b, \; \epsilon  \; \mathbb{R}, \; a>0  
$$
"""
st.markdown(texto2)

st.header("Transformada de Wavelet discreta (DWT)")
texto3=r"""
La transformada de Wavelet discreta se obtiene tomando valores discretos de los parámetros a y b para la transformada de Wavelet continua:
$$
a =2^{j}
$$
$$
b=k2^{j}
$$
por tanto:
$$
\psi_{j,k}(t)=(2^{j})^{-1/2} \psi \left ( \frac{t-k2^{j}}{2^{j}} \right )=2^{-j/2} \psi \left( 2^{-j}t-k\right), \; con \; j,k \; \epsilon \;\mathbb{Z}
$$

De este modo se puede calcular la transformada de Wavelet discreta como:
$$
DWT(j,k) = \int f(t) \psi_{j,k}^{*}(t) dt, \; j, \; k \; \epsilon \; \mathbb{Z}
$$

Una función $f(t)$ de soporte finito puede ser reconstruída como suma de los coeficientes Wavelets $DWT(j,k)$ multiplicados por las funciones $ \psi_{j,k}(t)$ (si forman una base ortonormal):

$$
f(t)=\sum_{j,k}DWT(j,k)\psi_{j,k}(t)= \sum_{j,k}DWT(j,k)2^{-j/2}\psi(2^{-j}t-k)
$$


Dado un **nivel de resolución M**, se puede representar la unción $f(t)$ como:

$$
f(t) = \sum_{k}2^{-M/2}c_{M}(k) \phi(2^{-M}t-k) + \sum_{j=1}^{M} \sum_{k}2^{-j/2}d_{j}(k)\psi(2^{-j}t-k)
$$

donde $\phi(t)$ es la función escala. Las funciones Wavelet son las encargadas de representar los detalles de la función $f(t)$, mientras que las funciones de escala realizan una aproximación.

La transformada Wavelet se implementa como un árbol jerárquico de filtrado organizado en niveles de descomposición.
"""
st.markdown(texto3)

st.markdown("![ArbolFiltrado.jpg](data:image/jpeg;base64,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oAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigApMihvumvGZ/G3iqLxHr3gu2RLjxM1yJdNu3jCwxWLjPmvgYJTlfclevNAHs9FVtMing0+CK7uDdXCIBJOUCeY2OW2jgZ9KsZoAWuF+Of/JIfFn/AF4SV3Wa4X45/wDJIfFn/XhJQB3KfcX6UtNj+4PpTqACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAD04rhfhjbRT3HiTW5FD315qlxC0rcsI4XMaID2UBc49STXc5zXGfCk/wDEi1L/ALC9/wD+lD0AdpRRRQAUVRuNWsrbVLPTZ51S9vFd4IjnMgTG4j6bh+dNsta0++1XUNNtblZL2w2faYgDmLeMrn6gH8qANCiiigDhfiEiWOteEtWjQLdpqkdkZAMFoplZWQn0ztOPVRXciuI+Kv8AqfC//Yfs/wD0I12+aAFopM0ZoAG+6a5bV9QvNT1SXQ9BmNvLEFa+vQob7MrchFzwZGHTso5PUA9BqN9BY6ddXlwwWC3iaaRvRVBJ/QGsrwNavB4ctp7hAt5fD7bc+vmSfMQfpkL9AKALGj6Bp2js8lrbr9pkAEtzId80v+855P51r4rl/G9/dWE/h0WkzRi51WKCUD+JCrkj9BXT5xQBhax4ZsdQmN3GHsdSxhb60IjmHoCcfMPZgR7VH4b1e4e7l0fWtqaxbRhy6LtjuY84EqenPVf4SccggnZhvba5llht7iGWWI4kRJAxQ+hA6VgeOT/Z8On62oAfT7lPMPfyJGCSD6AEN/wAUAdRRSDijNAC0UmaTevqKAHUUZrP1fWbDR1tTqNwIRdXEdpDlSd8rnCLwO5/CgDQooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAKGvSahHpF02jQwTajsPkJO5SPd2LEc4rzO2+HWo6TdaBr1jdRXXimK6Mur3czFPtcUgAkQcHAXC7Bjjb7163RigBFrG8Q+GrDX2gN+14DDu2/Z7uSD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)")

texto4=r"""
En un **banco de filtros** la señal de entrada se descompone en dos señales: una con frecuencias bajas (aproximación de la señal original) y otra con frecuencias altas (se asocia a los detalles). A la salida de los filtros se aplica un diezmado para descartar una de cada dos muestras. Si no se aplicase este diezmado a la salida del filtro se tendría el doble de datos que a la entrada.

El árbol jerárquico de filtrado consiste en implementar una serie de bancos de filtros. A la salida de cada etapa $j$ del filtrado jerárquico se obtienen los coeficientes de aproximación  $c_{j}$  y los coeficientes de detalle $d_{j}$ . La entrada para la siguiente etapa ($j+1$) son los coeficientes de aproximación de la etapa j ($c_{j}$). De esta forma se obtiene un vector de coeficientes Wavelet $c_{M}, \; d_{M}, \; d_{M−1}, \; ⋯, \: d_{1}$.

A partir de los coeficientes $c_{M}, \; d_{M}, \; d_{M−1}, \; ⋯, \: d_{1}$, es posible reconstruir la señal original mediante otra estructura de filtros similar a la descrita anteriormente.

La transformada de Wavelet se puede utilizar, además de para la compresión de señales, para reducir el ruido de una señal. En primer lugar, se seleccionaría una función Wavelet madre y se aplica la transformada Wavelet a la señal inicial con ruido. A continuación se realiza un "recorte de coeficientes" y se eliminan los componentes de ruido. Finalmente, se realiza la transformación Wavelet inversa para obtener la señal inicial pero sin ruido.

Una de las Wavelet más simples que se pueden definir es la **haar**.
"""
st.markdown(texto4)

st.header("Familia Haar: función escalón")

texto5=r"""
Se define la función wavelet de Haar, $\psi(t)$ y la función escalada $\phi(t)$:

$$
\psi(t)= \left\{ \begin{array}{rlc} 1 & si & 0 \leq t < 1/2  \\ 
-1 & si & 1/2 \leq t < 1 \\  0 & & en \: otro \: caso \end{array} \right.
$$ 

$$
\phi(t)= \left\{ \begin{array}{rlc} 1 & si & 0 \leq t < 1   
\\  
0 & & en \: otro \: caso \end{array} \right.
$$
"""

st.markdown(texto5)

p=[-0.5,0,0,0.5,0.5,1,1,1.5]
phi=[0,0,1,1,-1,-1,0,0]
psi=[0,0,1,1,1,1,0,0]

fig1=plt.figure(figsize=(12,6))
f1= fig1.add_subplot(221)
f1.plot(p,phi,drawstyle='steps-pre')
f1.set_xlabel('t')
f1.set_ylabel(r'$\psi(t)$')
f1.set_title('Función wavelet')
f1.grid(0.5)

f2= fig1.add_subplot(222)
f2.plot(p,psi,drawstyle='steps-pre')
f2.set_xlabel('t')
f2.set_ylabel(r'$\phi(t)$')
f2.set_title('Función escalada')
f2.grid(0.5)

with st.expander("Ver gráficas"):
    st.pyplot(fig1)

plt.cla()
plt.close(fig1)
st.divider()


st.subheader("Selección de datos", divider="red")

ejemplo = st.radio(
    "**Seleccion el conjunto de datos**",
    ["Datos de ejemplo", "Mis propios datos"],
    captions=[
        "Datos de fusión termonuclear",
        "Datos subidos en el apartado de Señales digitales",
    ],
)

if ejemplo == "Datos de ejemplo":
    datos=getDatosTF()
    nombreSeries=getNombreSeriesTF()
    tiposSeries=getTipoSeriesTF()  
else:
    if getExternos() is not None:
        datos=getExternos()
        nombreSeries=list(datos.columns)[1:]
        tiposSeries=np.unique(list(map(lambda x: x[0:x.find('_')] if '_' in x else x,nombreSeries)))

    else:
        st.page_link("./PaginaWeb/senhales/senhales.py", label="PULSE en el enlace para subir un archivo válido al final de la página de señales digitales")
        datos=getDatosTF()
        nombreSeries=getNombreSeriesTF()
        tiposSeries=getTipoSeriesTF()  

  
with st.expander("Ver datos"):
    st.write(datos)
#######################################################
#datos=getDatosTF()
#nombreSeries=getNombreSeriesTF()
#tiposSeries=getTipoSeriesTF()
######################################################


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

    serie=pd.Series(datos[optionS])
    serie.index=datos['t']

with col2:
    dibujarSerie(serie)

texto6=r"""
PyWavelets es un software para el cálculo de la transformada de Wavelet de código abierto para Python. Mediante la función $\texttt{dwt}$ se puede calcular los coeficientes de aproximación y detalle para un único nivel de descomposición
"""
st.write(texto6)

st.subheader("Elección de la wavelet madre", divider="red")

col1, col2 = st.columns(2)
with col1:
    optionF=st.selectbox(
        "Seleccione una familia de señales",
        options=pywt.families(), 
        index=0       
    )
    
    optionW=st.selectbox(
        "Seleccione Wavelet",
        options=pywt.wavelist(optionF),
        index=0
    )
with col2:

    dibujarWavelet(optionW)

st.subheader("Coeficientes de aproximación y detalle", divider="red")
digujarCoeficientes(serie,optionW)

st.subheader("Suavizado", divider="red")

texto7=r"""
La función $\texttt{wavedec}$ de la librería PyWavelets permite fijar el nivel de descomposición M.

Si se hacen igual a cero todos los coeficientes de detalle $d$ para los niveles de resolución, es decir, se aíslan los coeficientes wavelet correspondientes a las componentes de alta frecuencia y se calcula la transformada de wavelet discreta inversa se obtiene:
"""

st.write(texto7)

col21, col22 = st.columns(2)
with col21:
    texto8="""
    Fíjese un nivel de descomposición y véase como cambia la transformada Wavelet inversa al convertir en ceros todos los coeficientes de detalle para todos los niveles.
    """
    st.write(texto8)
    mSelec=st.slider("Nivel de descomposición: ", min_value=1,max_value=math.trunc(math.log2(len(serie.index)))-1 ,value=math.trunc(math.log2(len(serie.index))/2))

    texto9="""
    **Número de coeficientes de aproximación** :
    """
    
with col22:
    dibujarSerie(serie)


n=dibujarCAeInv(serie,optionW,mSelec)
st.write(texto9+str(n))

