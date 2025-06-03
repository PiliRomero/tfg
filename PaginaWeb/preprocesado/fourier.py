import streamlit as st
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
from PaginaWeb.funciones.fun import *

#################################
# Transformada de Fourier
#################################

st.title("Transformada de Fourier")

#################################
# Teoría
#################################

st.markdown("La transformada de Fourier descompone la señal en componentes sinusoidales con diferentes frecuencias, es decir, permite representar las señales en el **dominio de frecuencias**")
st.header("Transformada de Fourier en tiempo continuo")
texto1=r"""
Si $x(t)$ es una señal periódica (con período fundamental T) para qla que se cumplen las condiciones de Dirichlet entonces se puede descomponer utilizando el **desarrollo en series de Fourier** mediante un sumatorio de fasores:

$$
x(t) = \sum_{k=-\infty }^{+\infty }a_{k}e^{jkw_{0}t}, \; t \: \epsilon \:  \mathbb{R} 
$$ 

donde los coeficientes $a_{k}$, denominados coeficientes espectrales, se determinan mediante:

$$
a_{k} = \frac{1}{T} \int_{-T/2}^{T/2}x(t)e^{-jkw_{0}t}dt
$$
Las condiciones de Dirichlet que se enumeran a continuación contituyen un conjunto de condiciones suficientes para que una señal periódica $x(t)$ se pueda descomponer mediante la transformada de Fourier:


*   La señal ha de ser periódica
*   La señal ha de ser continua a trozos y tener un número finito de máximos y mínimos en $[-T/2,T/2]$
*   $\int_{-T/2}^{T/2}\left | x(t) \right |dt \; < \;\infty$


Si se pretende trabajar con series no periódicas no se puede aplicar el desarrollo de series de Fourier, se debe utilizar la **transformada de Fourier** de una señal continua:


* Transformada de Fourier inversa: $$x(t)=\frac{1}{2\pi}\int_{-\infty}^{+\infty}X(ω)e^{jwt}dω, \; t \; \epsilon \; \mathbb{R}  $$
* Transformada de Fourier directa: $$X(w) = \int_{-\infty}^{+\infty} x(t) e^{-jwt} dt, \; \omega \; \epsilon \; \mathbb{R} $$



Se puede demostrar que la transformada de Fourier existirá si es de energía finita:

$$
\int_{-\infty}^{+\infty}  \left | x(t)  \right | ^{2}dt < \infty
$$

Para poder analizar las señales de tiempo discreto no se puede utilizar la transformada de Fourier vista en el este apartado.
"""
st.markdown(texto1)

st.header("Transformada de Fourier en tiempo discreto")
texto2 = r"""
* Transformada inversa de Fourier en tiempo discreto:
$$
x(n) = \frac{1}{2π}\int_{0}^{2π} X(ω)e^{jωn}dω, \; n \; \epsilon \; \mathbb{Z} 
$$
* Transformada de Fourier en tiempo discreto:
$$
X(ω) = \sum_{n=-∞}^{+∞} x(n) e^{-jωn}, \; \omega \; \epsilon \; [0,2\pi]
$$
"""
st.markdown(texto2)

st.header("Transformada discreta de Fourier")
texto3=r"""
Al calcular la transformada de Fourier de una serie lo que se obtiene es una función compleja de variable real ($X(ω)$ es una función continua de $ω$), sin embargo, al aplicar la transformada discreta de Fourier a una serie, lo que se va a conseguir es una serie de valores discretos. En efecto,  se va a representar la serie $x(n)$ mediante muestras del espectro $X(ω)$.


* Transformada discreta de Fourier 
$$
X(k) = \sum_{n=0}^{N-1}x(n)e^{-j\frac{2π}{N}kn}, \; k=0,1, \cdots , N-1
$$
* Transformada inversa: 
$$
x(n) = \frac{1}{N}\sum_{k=0}^{N-1}X(k)e^{j\frac{2π}{N}kn}, \; n=0,1, \cdots , N-1
$$

Calcular la transformada discreta de Fourier puede resultar computacionalmente muy costoso (en torno a $N(N-1) + (N-1)^{2}$ operaciones).
El algorimo de la transformada rápida de Fourier (FFT) permite reducir la complejidad de la transformada discreta de Fourier de $O(N^{2})$ a $O(N \: logN)$.

Para poder aplicar la transformada discreta de Fourier de N puntos se debe verificar que $x(n)=0 \; ∀ \;n<0,\; n>N$
"""
st.markdown(texto3)

#########################################################################
# Selección del conjunto de datos (ejemplo o facilitados por el usuario)
#########################################################################

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

#################################
# Selección de la señal
#################################

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

#################################
# Transformada rápida de Fourier
#################################

st.subheader("Representación de los coeficientes de la FFT", divider="red")

texto4="""
Debe tenerse en cuenta que los valores de la transformada discreta de Fourier son números complejos, por lo que es necesario a la hora de dibujar la serie transformada, representar tanto su parte real como su parte imaginaria.
"""
st.markdown(texto4)

dibujarTransformada(serie)

texto5=r"""
Otra alternativa es representar el módulo y la fase de las componentes de la transformada discreta de Fourier. Basta recorda que, dado un número complejo $a + bi$, el módulo viene dado por $\sqrt{a^{2}+b^{2}}$ y la fase por $arco tangente (\frac{b}{a})$
"""
st.write(texto5)
dibujarTransformadaMF(serie)

#################################
# Suavizado de señales
#################################

st.subheader("Suavizado de señales", divider="red")

texto6="""
Se puede pensar en hacer igual a cero las componentes de la transformada discreta de Fourier cuyo módulo (hay que recordar que las componentes de la transformada son números complejos) se encuentren por debajo de un cierto umbral y a continuación calcular la transformada inversa de Fourier. Esta técnica puede utilizarse para quitar ruido a la señal original.

En el siguiente menú se pueden elegir distintos umbrales para el módulo más pequeño de la transformada discreta de Fourier y ver como cambia la representación gráfica de la inversa de Fourier conforme se modifica este umbral.
"""
st.write(texto6)


moduloSelec = st.slider("Seleccione el módulo: ", min_value=0.0,max_value=moduloMaximo(serie) ,value=percentil(serie))

dibujarSerieInv(serie, modulo=moduloSelec)


texto7="""
A continuación se puede es posible variar el número elementos no nulos de la transformada discreta de Fourier para seguidamente calcular la inversa de Fourier y representarla en el dominio del tiempo.
"""
st.markdown(texto7)

numeroSelec = st.slider("Número: ", min_value=1,max_value=len(serie.index) ,value=max(int(len(serie.index)*0.01),2))
dibujarSerieInv(serie,numero=numeroSelec)

st.subheader("Funciones ventana", divider="red")
texto8=r"""
En los ejemplos anteriores se ha utilizado el algoritmo FFT.
Al aplicar directamente la transformada discreta de Fourier no se ha tenido en cuenta la hipótesis de que $x(n)=0
\; ∀ \;n<0,\; n>N$. 
Para lograr que se verifique esta condición se puede multiplicar la señal por una **función ventana** antes de aplicarle la transformada rápida de Fourier.
"""
st.markdown(texto8)

col12, col22 = st.columns(2)

with col12:
    dibujarSerie(serie,ventana=True)

dibujarTransformada(serie,ventana=True)


texto9="""
En el siguiente ejemplo parte de la serie digital selecciona y se le aplica la función ventana (en este caso la ventana Hamming) antes de aplicarle la transformada rápida de Fourier. Después se selecciona un valor umbral para módulo y se calcula la inversa de Fourier para esta nueva señal. Por último se divide la señal inversa entre la función ventana para tratar de reconstruir la señal original.
"""
st.markdown(texto9)

moduloSelec2 = st.slider("Seleccione el módulo: ", min_value=0.0,max_value=moduloMaximo(serie,ventana=True) ,value=percentil(serie,ventana=True))

dibujarSerieInv(serie,modulo=moduloSelec2,ventana=True)
