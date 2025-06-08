import pandas as np
import pywt
import math
from sklearn.cluster import KMeans
from PaginaWeb.funciones.fun import *

########################################
# Determinación del número de clústeres
########################################

st.title("Número de clústeres")

texto1="""
En esta página se analizan dos métodos que permiten determinar cuál es el número óptimo de clústeres en una agrupamiento:
"""
#################################
# Teoría
#################################

st.markdown(texto1)
st.header("El método del codo")
texto2="""
El **método del codo** consiste en aplicar el método de agrupamiento para diferentes números de clústeres (k) y calcular la suma de las distancias al cuadrado de cada instancia al centroide del clúster. Al aumentar el número de clústeres la suma de distancias disminuye, pero a partir de un determinado valor de k, esta disminución es despreciable. Se dibuja una gráfica en donde se representa en el eje de abscisas el número de clústeres k y en el de ordenadas la suma de las distancias al cuadrado. El k óptimo se corresponde con el valor de la abscisa para el cual la disminución de la suma de distancias deja de ser significativa.
"""
st.markdown(texto2)

st.header("El método Silhouette")
texto3=r"""
El **método Silhouette** determina el número óptimo de clústeres maximizando la media del coeficiente silhouette. Este coeficiente indica en qué media es similar una instancia al clúster al que se ha asignado en comparación con los restantes clústeres.

El valor del coeficiente silhouette varía entre -1 (la asignación no es buena) y 1 (la observación se ha asignado al clúster correcto).  

Para cada ejemplo de entrenamiento se calcula el coeficiente silhouette ($s_{i}$) como:
$$
s_{i} = \frac{b_{i}-a_{i}}{max(a_{i},b_{i})}
$$
donde:


*   $a_{i}$ es el promedio de las distancias de la instancia i al resto de instancias pertenecientes al mismo clúster.
*   $b_{i}$ es la distancia mínima entre la instancia i y el resto de clústeres. Para calcular la distancia entre una instancia y un clúster se calcula la media de las distancias de esta instancia a cada una de los elementos que componen el clúster.
"""
st.markdown(texto3)

#################################
# Selección del conjunto de datos
#################################

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

with st.expander("Ver coeficientes de aproximación" + cadena):
    st.write(datosP)

st.divider()

###################################################################
# Determinar número óptimo de clústeres
###################################################################

texto4="""
A continuación se prueba a obtener el número "óptimo" de clústeres para el agrupamiento por k medias aplicanco el método codo y el método Silhouette.
"""
st.write(texto4)
st.subheader("Método del codo", divider="red")
metodoCodo(datosP)
st.subheader("Método Silhouette", divider="red")
k=optimoSilhouette(datosP)
st.write("Número óptimo de clústeres: " + str(k))
kmedias=KMeans(n_clusters=k,init='random')
clasesKm=kmedias.fit_predict(datosP)
dibujarClusters(k,datosP,clasesKm)
