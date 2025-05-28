import streamlit as st
import pandas as np
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import scale
from PaginaWeb.funciones.fun import *

st.title("Agrupamiento jerárquico")

texto1=r"""
El agrupamiento jerárquico es una técnica de aprendizaje automático no supervisado que agrupa las instancias no etiquetadas en clases. El objetivo es construir una jerarquía de
clases de nodos (clústeres) de modo que cada clase contenga un subconjunto de instancias que han de ser similares entre sí. 

A grandes rasgos los pasos que sigue el agrupamiento jerárquico son:
1. Se crea una clase por cada instancia
2. Se buscan las dos clases más parecidas según la medida de similitud establecida. En caso de que todas las variables sean continuas se suele trabajar con distancias, siendo la distancia euclídea la más utilizada.
3. Se crea una nueva clase que agrupa a las dos clases seleccionadas en el punto 2.
4. Se calcula el **centroide** de cada clase, que es un punto equidistante de las instancias pertenecientes a cada clase.  
5. Volver al paso 2 hasta que únicamente quede una clase (nodo raíz)

Si se desea realizar un agrupamiento jerárquico para los ejemplos recogidos en la figura que se muestra a continuación: """
st.write(texto1)
st.image("./PaginaWeb/imagenes/aglomerativo.png")
texto11=r"""en primer lugar, se agrupan las clases H y F por ser las más próximas. A continuación se forma un nuevo clúster con las observaciones G y J. En
el siguiente paso se añade E al clúster formado por G y J. En la iteración posterior se unen B y D para formar un nuevo clúster. Así sucesivamente
hasta que los 10 ejemplos formen parte del mismo clúster. 

Como se puede observar se trata de un algoritmo de "abajo hacia arriba".
Para calcular la distancia entre dos clases existen diferentes técnicas, entre las que cabe destacar:

*   **Encadenamiento simple**: la distancia entre dos clases (criterio de enlace) se calcula como la distancia ente los dos puntos más próximos de las clases.
*   **Encadenamiento completo**: se toma la distancia entre los dos puntos más alejados de las clases.
*   **Método de Ward**: define la distancia entre dos clases $A$ y $B$ como la magnitud en la que se incrementa la suma de cuadrados de los errores cuando se unen estas clases. Es decir, el método de Ward trata de maximizar la homogeneidad dentro de las clases:
$$ 
\Delta (A,B) = \sum_{i \epsilon A \cup B} \left\| x_{i} - c_{A\cup B}\right\|^{2} - \sum_{i \epsilon A} \left\| x_{i} - c_{A}\right\|^{2} - \sum_{i \epsilon B} \left\| x_{i} - c_{B}\right\|^{2}
$$
donde $c_{A}$ y $c_{B}$ con los centroides de las clases A y B y $c_{A \cup B}$ es el centroide tras unir ambas clases.
"""
st.markdown(texto11)
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
Para aplicar la técnica de agrupamiento jerárquico con el fin de tratar de identificar formas de onda completas, 
en primer lugar se realiza el **preprocesamiento** de las señales. En los siguientes ejemplos se optó por aplicar la 
transformada de Wavelet, por poderse aplicar a señales estacionarias o no estacionarias. Se fija un nivel de descomposición M común para todas las señales. Cada instancia, o ejemplo 
de entrenamiento del algoritmo, estará formada por los coeficientes de aproximación para este nivel de descomposición. 
De este modo se representará cada una de las señales originales por un número de coeficientes que dependerá de la profundidad M.
 A mayor profundidad la reducción de la dimensionalidad va a ser más significativa, pero el ajuste va a ser peor.
"""
st.write(texto2)
st.subheader("Selección de wavelet y nivel de descomposición", divider="red")
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
with st.expander("Ver coeficientes de aproximación" + cadena):
    st.write(datosP)

st.subheader("Selección de distancia de enlace y medida", divider="red")

enlace=linkage()
medida=medida(enlace)

st.subheader("Dendrograma", divider="red")

texto3="""
El **dendrograma** es un árbol que permite visualizar como se agrupan las instancias en los diferentes niveles.  
Los nodos hoja del árbol se corresponden con las instancias individuales (transformadas Wavelet) y el nodo raíz es la 
clase que agrupa a todas las instancias.
"""

st.write(texto3)
cluster_dist = AgglomerativeClustering(n_clusters=2,metric=medida,linkage=enlace,compute_distances=True)
cluster_dist.fit(datosP)


texto4="""
El dendrograma resulta de utilidad para determinar las instancias que pertenecen a cada clase. Para ello se traza una línea horizontal y se poda el dendrograma descartando los clústeres formados por debajo de la línea trazada.
"""
st.write(texto4)

alturaSelect=st.slider("Altura de corte del dendrograma: ", min_value=0.0,max_value=max(cluster_dist.distances_) ,value=max(cluster_dist.distances_)/2)

dibujarDendograma(cluster_dist,alturaSelect,medida,enlace)

st.subheader("Métrica de evaluación", divider="red")

texto5="""
A continuación se pide introducir el número de clústeres para el agrupamiento jerárquico y se dibujan las señales agrupadas en función del clúster al que pertenecen. Además, se calcula el índice de  Davies Bouldin. 

El **índice de Davies Douldin** es una métrica de evaluación interna (sin referencias externas) que trata de medir la bondad de la agrupación obtenida. 

Este índice se calcula como el cociente de las medias de las distancias dentro del clúster y las distancias entre los clústeres.

Cuanto más próximo a cero esté el valor del índice de Davis Bouldin mejor será la bondad del ajuste (mayor cohesión dentro de los clústeres y mayor separación entre los diferentes clústeres). 
"""

st.write(texto5)
st.subheader("Representación gráfica de los grupos", divider="red")

nCSelect=st.slider("Número de clústeres: ", min_value=2,max_value=len(datosP) ,value=2)

clusterN = AgglomerativeClustering(n_clusters=nCSelect,metric=medida,linkage=enlace,compute_distances=True)
clases=clusterN.fit_predict(datosP)
etiquetas=np.unique(clases)

dabo=davies_bouldin_score(datosP, clases)
st.write("Índice de Davies Bouldin: " + str(round(dabo,3)))

dibujarClusters(nCSelect,datosP,clases)

st.subheader("Reducción de la dimensionalidad", divider="red")
#componentesPrincipales(datosP)
#componentesPrincipales2(datosP)
datosCP=componentesPrincipales2(datosP)
if normalizar:
    datosCP=pd.DataFrame(scale(datosCP),index=seriesSelect,columns=['pc1','pc2'])
clasesCP=clusterN.fit_predict(datosCP)
etiquetasCP=np.unique(clasesCP)

texto6="""Para poder visualizar los clústeres se puede reducir la dimensionalidad de los datos a los que se les aplican los algoritmos aglomerativos a dos dimensiones. 
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