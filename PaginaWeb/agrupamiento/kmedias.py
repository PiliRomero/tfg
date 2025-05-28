import streamlit as st
import pandas as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from PaginaWeb.funciones.fun import *

st.title("k-medias")

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

#######################################################
#datos=getDatosTF()
#nombreSeries=getNombreSeriesTF()
#tiposSeries=getTipoSeriesTF()
######################################################

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

dabo=davies_bouldin_score(datosP, clases)
st.write("Índice de Davies Bouldin: " + str(round(dabo,3)))

dibujarClusters(nCSelect,datosP,clases)

#componentesPrincipales2(datosP)
datosCP=componentesPrincipales2(datosP)
if normalizar:
    datosCP=pd.DataFrame(scale(datosCP),index=seriesSelect,columns=['pc1','pc2'])
clasesCP=kmedias.fit_predict(datosCP)
etiquetasCP=np.unique(clasesCP)

texto6="""
Para poder visualizar los clústeres se puede reducir la dimensionalidad de los datos a los que se les aplican los algoritmos aglomerativos a dos dimensiones. 
Para ello se utiliza el **análisis de componentes principales**."""
st.write(texto6)

texto7=r"""
La reducción de la dimensionalidad mejora el rendimiento de los algoritmos de aprendizaje automático al transformar el conjunto de datos en otro de menor dimensión, pero que sigue conservando las principales características de los datos originales. El análisis de componentes principales permite identificar patrones en los datos y facilita la visualización de los mismos, puesto que al considerar únicamente las dos primeras componentes, es posible representar las series en el plano euclídeo.

Se parte de un vector $X = (X_{1}, \dots, X_{p})^{t}$ de p dimensiones y se desea pasar a un vector reducido $Z = (X_{1}, \dots ,Z_{r})^{t}$, con $r<p$, obtenido a partir de $X$ y que contenga la máxima información (dispersión) que posee $X$. 

Se define la **primera componente principal** de $X$ como:
$$
Z_{1}=V_{1}^{t}X=V_{11}X_{1}+\dots+V_{p1}X_{p}, \: con \ V_{1}=(V_{11}, \dots, V_{p1})^{t} \:\epsilon \: \mathbb{R}^{p}
$$ 

tal que $Varianza(Z_{1})=max \left\{ varianza(V^{t}X) \: / V \: \epsilon \: \mathbb{R}^{p}, \: V^{t}V=1 \right\}$

Se puede demostrar que la primera componente principal adopta la forma $Z_{1}=V_{1}^{t}$ siendo $\lambda_{1}$ es el mayor autovalor de $\Sigma=D(X)=E(XX^{t})-(E[X])(E[X])^{t}$ 
y $V_{1}$ es un autovector de $\Sigma$ asociado a $\lambda_{1}$ de norma la unidad. 

Se definir la **segunda componente principal** de $X$ como una variable aleatoria 
$$
Z_{2}=V_{2}^{t}X=V_{12}X_{1}+\dots+V_{p2}X_{p}, \: con \ V_{2}=(V_{12}, \dots, V_{p2})^{t} \:\epsilon \: \mathbb{R}^{p}
$$

tal que $Varianza(Z_{2})=max \left\{ varianza(V^{t}X) \: / V \: \epsilon \: \mathbb{R}^{p}, \: V^{t}V=1, \; V_{1}^{t}V=0 \right\}$

La segunda componente principal de X adopta la forma $Z_{2} = V_{2}^{t}X$, siendo $\lambda_{2}$ el segundo mayor autovalor de $\Sigma$ y $V_{2}$ un autovalor de $\Sigma$ asociado a
$\lambda_{2}$ de norma uno. 

Las p componentes principales de X adoptan la forma:
$$
Z_{j} = V_{j}^{t}X, \: j\epsilon \left\{1, \dots , p\right\}
$$

siendo $\lambda_{1} \ge \dots \ge \lambda_{p} \ge 0$, los p autovalores ordenados de $\Sigma$ y $V_{1}, \dots, V_{p}$ sus autovectores asociados y de norma la unidad. 
"""
with st.expander("Ver explicación de componentes principales"):
    st.write(texto7)
with st.expander("Ver porcentaje de varianza explicada"):
    componentesPrincipales(datosP)
dibujarCluster2d(datosCP,clasesCP,nCSelect)
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