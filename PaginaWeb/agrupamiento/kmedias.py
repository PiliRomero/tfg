import streamlit as st
import pandas as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from PaginaWeb.funciones.fun import *

#################################
# Algoritmo de k medias
#################################

st.title("k-medias")

#################################
# Teoría
#################################

texto1="""
El algoritmo de aprendizaje automático no supervisado de k-medias es un método de partición, en contrapartida con el algoritmo de agrupamiento jerárquico, en el que se parte de una clase con instancias heterogéneas y se pretende dividirla en k grupos, donde el número de grupos (k) se fija de antemano. En este algoritmo no se crea una jerarquía de clases, sino que las k clases estarán en un único nivel.

El agrupamiento se realiza de forma que se haga mínima la suma de las distancias de cada instancia al centroide de la clase a la que pertenece, es decir, la partición se realiza de modo que cada instancia se asigna a la clase cuyo centroide está más próximo (o cuya similitud al centroide sea máxima). El algoritmo busca disminuir la variabilidad dentro de cada clase e incrementar la variabilidad entre distintas clases.

Los pasos que se han de seguir para aplicar este algoritmo se resumen en:

1.   Fijar el número de clases k.
2.   Escoger k centroides del conjunto de instancias, por ejemplo mediante selección aleatoria.
3. Asignar cada instancia a la clase o clúster cuya distancia al centroide sea menor (o la similitud sea máxima).
4. Recalcular los centroides (promedio de las instancias de cada grupo).
5. Repetir los pasos 3 y 4 hasta que la composición de las clases no se altere o se supere un número máximo de iteraciones establecido previamente.
"""
st.markdown(texto1)

#################################################################################
# Selección del conjunto de datos (ejemplo o proporcionados por el usuario)
#################################################################################

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
    if (getExternos() is not None) and (esPosibleAgrupamiento(getExternos())):
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



col1, col2 = st.columns(2)
with col1:
    optionTs=st.multiselect(
        "Seleccione un tipo de señal",
        options=tiposSeries,
        default=tiposSeries[0]    
    )

    seriesSelect=[]
    for ts in optionTs:
        for ns in nombreSeries:
            if ts in ns:
                seriesSelect.append(ns)


listaS=['t']
listaS.extend(seriesSelect)
datosSelect=datos[listaS]
with col2:
    st.write("Número total de series seleccionadas: " + str(len(seriesSelect)))
    if len(seriesSelect)>0:
        st.write("Número de observaciones para cada señal: " + str(len(datosSelect)))

    if(len(seriesSelect)<2):
        st.write("**Debe elegir al menos un tipo de series**")
        seriesSelect=[]
        ts=tiposSeries[0]
        for ns in nombreSeries:
            if ts in ns:
                seriesSelect.append(ns)
        listaS=['t']
        listaS.extend(seriesSelect)
        datosSelect=datos[listaS]   

###################################################################
# Selección y preprocesado de señales mediante transformada wavelet
###################################################################

texto2="""
A continuación se va a aplicar la técnica de k-Medias para lograr identificar formas de onda completas. Para ello en primer lugar se realiza el **preprocesamiento** de las señales aplicando la transformada de Wavelet. Se fija un nivel de descomposición M común para todas las señales. Cada instancia estará formada por los coeficientes de aproximación para este nivel de descomposición. De este modo se representará cada una de las señales originales por un número de coeficientes que dependerá de la profundidad M. A mayor profundidad la reducción de la dimensionalidad va a ser más significativa, pero el ajuste va a ser peor.
"""
st.write(texto2)



st.subheader("Selección de wavelet y nivel de resolución", divider="red")

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
    normalizar=st.checkbox("Normalizar coeficientes aproximación")

with col2:
    mSelect=st.slider("Nivel de resolución: ", min_value=1,max_value=math.trunc(math.log2(len(datosSelect)))-1 ,value=math.trunc(math.log2(len(datosSelect))/2))

    seriesP=obtenerSeriesPreprocesadas(datosSelect,seriesSelect,mSelect,optionW)
    datosP=pd.DataFrame(seriesP,index=seriesSelect,columns=["c"+str(indice) for indice in np.arange(len(seriesP[0]))+1])
    if normalizar:
        datosP=pd.DataFrame(scale(datosP),index=seriesSelect,columns=["c"+str(indice) for indice in np.arange(len(seriesP[0]))+1])
    st.write("Cada serie del conjunto seleccionada se representará por un total de " + str(len(datosP.columns)) + " coeficientes.")

cadena=''
if normalizar:
    cadena=' normalizados'

with st.expander("Ver coeficientes de aproximación"+cadena):
    st.write(datosP)

st.subheader("Métrica de evaluación", divider="red")

###################################################################
# Aplicación del algoritmo de k-media
###################################################################

texto33="""
A continuación se pide introducir el número de clústeres para el agrupamiento jerárquico y se dibujan las señales agrupadas en función del clúster al que pertenecen. Además, se calcula el índice de  Davies Bouldin. 

El **índice de Davies Douldin** es una métrica de evaluación interna (sin referencias externas) que trata de medir la bondad de la agrupación obtenida. 

Este índice se calcula como el cociente de las medias de las distancias dentro del clúster y las distancias entre los clústeres.

Cuanto más próximo a cero esté el valor del índice de Davis Bouldin mejor será la bondad del ajuste (mayor cohesión dentro de los clústeres y mayor separación entre los diferentes clústeres). 
"""

st.write(texto33)
st.subheader("Representación gráfica de los grupos", divider="red")

texto3="""El algoritmo de k-medias necesita que se fije de antemano el número de clústeres:"""

nCSelect=st.slider("Número de clústeres: ", min_value=2,max_value=len(datosP) ,value=2)
kmedias=KMeans(n_clusters=nCSelect,init='random')
clases=kmedias.fit_predict(datosP)

#############################################################
# Medidas de ajuste y representación gráfica de los clústeres
#############################################################

dabo=davies_bouldin_score(datosP, clases)
st.write("Índice de Davies Bouldin: " + str(round(dabo,3)))

dibujarClusters(nCSelect,datosP,clases)

##########################
# Componentes principales
##########################

datosCP=componentesPrincipales2(datosP)
if normalizar:
    datosCP=pd.DataFrame(scale(datosCP),index=seriesSelect,columns=['pc1','pc2'])
clasesCP=kmedias.fit_predict(datosCP)
etiquetasCP=np.unique(clasesCP)

texto6="""
Para poder visualizar los clústeres se puede reducir la dimensionalidad de los datos a los que se les aplican los algoritmos aglomerativos a dos dimensiones. 
Para ello se utiliza el **análisis de componentes principales**."""
st.write(texto6)

with st.expander("Ver explicación de componentes principales"):
    imprimirTextoComponentesPrincipales()
with st.expander("Ver porcentaje de varianza explicada"):
    componentesPrincipales(datosP)
dibujarCluster2d(datosCP,clasesCP,nCSelect)

with st.expander("Ver gráfico iterativo"):
    dibujarCluster2d_bis(datosCP,clasesCP,nCSelect)

###################################################################
# Comparación agrupamiento jerárquico y k-medias
###################################################################

st.subheader("Comparación con el método de agrupamiento jerárquico", divider="red")
texto8="""
Debe tenerse en cuenta que las distintas técnicas de aprendizaje por agrupamiento no tienen que proporcionar la misma partición para el
conjunto de datos de entrenamiento. En efecto, a continuación se dibuja el gráfico de dispersión para la primera y segunda
componente principal de los coeficientes de aproximación de la transformada Wavelet y se colorean en función del clúster al que pertenecen.
Se puede comprobar como existen instancias que en los modelos de agrupamiento jerárquico y k-medias forman parte de clústeres distintos. Para simplificar
en el método de agrupamiento jerárquico se ha utilizado la como función linkage la función ward.
"""
st.write(texto8)

nCSelect2=st.slider("Número de clústeres: ", min_value=1,max_value=len(datosP) ,value=2)
kmedias=KMeans(n_clusters=nCSelect2,init='random')
clasesKm=kmedias.fit_predict(datosCP)

etiquetasCP2=np.unique(clasesCP)
clusterN = AgglomerativeClustering(n_clusters=nCSelect2,linkage='ward')
clasesAg=clusterN.fit_predict(datosCP)


col21, col22 = st.columns(2)
with col21:
    st.subheader("Agrupamiento jerárquico")
    dabo=davies_bouldin_score(datosCP, clasesAg)
    st.write("Índice de Davies Bouldin: " + str(round(dabo,3)))
    dibujarCluster2d(datosCP,clasesAg,nCSelect2)
with col22:
    st.subheader("k-medias")
    dabo=davies_bouldin_score(datosCP, clasesKm)
    st.write("Índice de Davies Bouldin: " + str(round(dabo,3)))
    dibujarCluster2d(datosCP,clasesKm,nCSelect2)

  