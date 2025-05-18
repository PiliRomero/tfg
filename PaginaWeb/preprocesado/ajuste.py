import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from PaginaWeb.funciones.fun import *



st.title("Medidas de ajuste")

st.header("Error cuadrático medio")
texto1=r"""
$$
ECM = \frac{1}{N}\sum_{t=1}^{N}(x(n)- \widehat{x}(n))^{2}
$$
donde N es el número de muestras, $x(n)$ es el valor de la señal digital y $\widehat{x}(n)$ el valor estimado obtenido aplicando alguna de las transformadas vistas en los apartados anteriores.
"""
st.markdown(texto1)

st.header("Error absoluto medio")

texto2=r"""
$$
EAM = \frac{1}{N}\sum_{t=1}^{N} \left | x(n)-\widehat{x}(n) \right |
$$

Se puede comprobar como varían estas dos medidas de error en función del nivel de resolución M considerado para la transformada wavelet y el número N de coeficientes no nulos para la transformada
de Fourier. Es decir, se parte de una señal digital, se calcula la transformada wavelet con un nivel de resolución M dado y se desprecian los coeficientes de detalle.  A partir de los nuevos coeficientes wavelet se calcula la transformada wavelet discreta inversa y finalmente se obtienen el ECM y EAM. 
Para el análisis de Fourier se anulan los coeficientes de la transormada rápida de Fourier excepto los N que tengan el módulo más alto.
"""
st.markdown(texto2)

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
        st.page_link("./senhales/senhales.py", label="Debe subir un archivo válido al final de la página de señales digitales")
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
    st.pyplot(dibujarSerie(serie))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Transformada rápida de Fourier")
    nSelec=st.slider("N máximo ", min_value=1,max_value=min(100,len(serie)) ,value=min(100,len(serie))//2)

with col2:
    st.subheader("Transformada wavelet")
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

st.pyplot(dibujarECM(serie,optionW,nSelec))