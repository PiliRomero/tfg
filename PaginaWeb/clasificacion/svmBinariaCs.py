import streamlit as st
import pandas as np
import numpy as np
import pywt
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from PaginaWeb.funciones.fun import *

###################################################################
# SVM datos cuasi-separables linealmente
###################################################################

st.title("SVM para clasificación binaria")

st.header("Datos cuasi-separables linealmente")

##############
# Teoría 
##############

texto1=r"""
Cuando el conjunto de datos de partida no es linealmente separable, se introducen las **variables holgura**
 $\xi_{i}, \; i=1, \dots, \; n$, que van a permitir que existan errores de clasificación para algunos datos de partida. 

En los datos cuasi-separables linealmente las restricciones del problema primal se relajan:

$$
y_{i}(\omega^{t}x_{i} + b) \geq 1 - \xi_{i}, \; i=1, \dots, \;n
$$

Por ejemplo, en la siguiente figura existe dos datos que no son linealmente separables y que no están correctamente clasificados pues caen en el lado incorrecto del margen del hiperplano.  
"""
st.write(texto1)

st.image("./PaginaWeb/imagenes/hiperplano2.jpg")

texto2=r"""

* Si $\xi_{i}=0$ el dato $(x_{i},y_{i})$ es separable
* Si $0<\xi_{i} < 1$ el dato no es linealmente separable pero está clasificado correctamente
* Si $\xi_{i} > 1$ el dato no es separable y está mal clasificado.


La suma de las variables de holgura $\sum_{i=1}^{n} \xi_{i}$ permite medir el coste asociado a los datos que no son linealmente separables.


Además de relajar las restricciones del problema primal es necesario incluir los errores de clasificación 
que se cometen con el hiperplano de separación en la función a minimizar. 
Así, la función objectivo pasa de ser $\frac{1}{2}\left\| \omega \right\|^2$ a:
$$\frac{1}{2}\left\| \omega \right\|^2 + C \sum_{i=1}^{n} \xi_{i}$$
donde C es el **parámetro de regularización**.
C es una constante que deberá determinar el usuario.


* Si C es muy grande, los valores de $\xi_{i}$ han de ser muy pequeños por lo que el margen del hiperplano será estrecho pero puede producirse un **sobreajuste** a los valores de entrenamiento. Sólo se busca separar los datos sin tener en cuenta el margen del hiperplano.   

* Si C es pequeño se permiten valores de $\xi_{i}$ más grandes, lo que aumenta el ancho de margen óptimo, lo que puede provocar que se lleguen a admitir dentro del margen a datos mal clasificados.


El problema de optimización consistirá por tanto en:

$$
 min \; \frac{1}{2}\left\| \omega \right\|^2 + C \sum_{i=1}^{n} \xi_{i}\\
 s.a.: \; y_{i}(\omega^{t}x_{i}+ b) + \xi_{i} -1 \geq 0, \; i=1, \dots, \; n 
$$


Utilizando los multiplicadores de Lagrange de manera análoga a SVM para datos separables linealmente , se llega al siguiente problema dual:

$$
\underset{\alpha}{max} \; \sum_{i=1}^{n} \alpha_{i} - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_{i} \alpha_{j}y_{i}y_{j} x_{i}^{t}x_{j}\\
s.a.: \; \sum_{i=1}^{n} \alpha_{i}y_{i} = 0 \\
0 \leq \alpha_{i} \leq C, \: i=1, \dots , \; n
$$


Una vez que se obtiene la solución óptima $\hat{\alpha}$ del problema dual  se obtiene $\hat{\omega} = \sum_{i=1}^{n}\hat{\alpha_{i}} y_{i}x_{i}$ y a continuación  $\pi = \left\{ x\; \epsilon\;  \mathbb{R}^{p}  \; | \; \sum_{i=1}^{n} \hat{\alpha_{i}}y_{i}x^{t}x_{i} + \hat{b}=0 \right\}$. 
Para determinar el término independiente del hiperplano se consideran los coeficientes de Lagrange $\alpha_{i}$ tales que $0 < \alpha_{i} < C$. El ejemplo de entrenamiento correspondiente $(x_{i},y_{i})$ será un vector soporte y permitirá calcular $\hat{b}$ como $\hat{b} = y_{i} - (\hat{\omega})^{t}x_{i}$,
 o realizando el promedio para todos los vectores soporte. 

 Hay dos tipos de vectores soporte con $\hat{\alpha_{i}} \ne 0$:


* Si $0 < \hat{\alpha_{i}} < C$ se corresponden con datos situados en la frontera de los márgenes del hiperplano.
* Si $\hat{\alpha_{i}}=C$ el ejemplo $x_{i}$ no es separable

"""
st.write(texto2)

###################################################################
# Selección de datos
###################################################################

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
    if (getExternos() is not None) and (esPosibleClasBin(getExternos())):
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


###################################################################
# Selección y preprocesado de señales mediante transformada wavelet
###################################################################

texto2="""
A continuación se muestra un desplegable para seleccionar las señales a las que se pretende aplicar los algoritmos SVM para clasificación binaria.
Debe seleccionar dos tipos de series distintas.
"""
st.write(texto2)

col1, col2 = st.columns(2)
with col1:
    sDefecto=tiposSeries[:2]
    optionTs=st.multiselect(
        "Seleccione un tipo de señal",
        options=tiposSeries,
        default=sDefecto   
    )

    seriesSelect=[]
    for ts in optionTs:
        for ns in nombreSeries:
#           if ts in ns:
            if ts==ns[0:ns.find('_')]:
                seriesSelect.append(ns)

listaS=['t']
listaS.extend(seriesSelect)
datosSelect=datos[listaS]
with col2:
    st.write("Número total de series seleccionadas: " + str(len(seriesSelect)))
    if len(seriesSelect):
        st.write("Número de observaciones para cada señal: " + str(len(datosSelect)))

    if(len(optionTs)!=2):
        st.write("**Debe elegir dos tipos se series distintos**")
        seriesSelect=[]
        for ts in tiposSeries[:2]:
            for ns in nombreSeries:
#           if ts in ns:
                if ts==ns[0:ns.find('_')]:
                        seriesSelect.append(ns)
        listaS=['t']
        listaS.extend(seriesSelect)
        datosSelect=datos[listaS]

st.subheader("Preprocesamiento", divider="red")
texto6="""
Se realiza el **preprocesamiento** de las señales mediante la transforma Wavelet. 
Se fija un nivel de descomposición M común para todas las señales. Cada instancia, o dato 
de entrenamiento del algoritmo, estará formada por los coeficientes de aproximación para este nivel de descomposición. 
"""
st.write(texto6)
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

###########################
# Componentes principales
###########################

st.subheader("Reducción de la dimensionalidad", divider="red")

datosCP=componentesPrincipales2(datosP)
if normalizar:
    datosCP=pd.DataFrame(scale(datosCP),index=seriesSelect,columns=['pc1','pc2'])


texto4="""
Para poder visualizar los vectores soporte y el hiperplano de separación se puede  reducir la dimensionalidad de los datos a dos dimensiones. 
Para ello se utiliza el **análisis de componentes principales**."""
st.write(texto4)

with st.expander("Ver explicación de componentes principales"):
    imprimirTextoComponentesPrincipales()
with st.expander("Ver porcentaje de varianza explicada"):
    componentesPrincipales(datosP)

cadena=''
if normalizar:
    cadena=' normalizadas'
with st.expander("Ver primera y segunda componentes principales"+cadena):
    st.write(datosCP)

###################################################################
# División de los datos en entrenamiento y test
###################################################################

st.subheader("Datos para el entrenamiento y para el test", divider="red")
clases,x,y=etiquetar(datosCP)
col1, col2 = st.columns(2)
with col1:
    st.write("**Número de señales de cada clase**")
    for c in clases:
        st.write("Clase "+c+": "+ str(len(y[y.values==clases.index(c)])) + " señales, etiqueta: " + str(clases.index(c)))

with col2:
    with st.expander("Ver distribución:" + cadena):
        dibujarDist(y,clases)


if "aleatorio" not in st.session_state:
    st.session_state.aleatorio = np.random.randint(9999)


if st.button("Volver a barajar", type="primary"):
    st.session_state.aleatorio = np.random.randint(9999)



xEntrenamiento, xTest, yEntrenamiento, yTest = train_test_split(x,y.values.reshape(-1,1),random_state=st.session_state.aleatorio,train_size=0.8,shuffle=True)

texto3="""
El siguiente paso es dividir el conjunto de señales en datos de entrenamiento y test. Se tomará el 80% de las instancias para el entrenamiento del modelo y el 20% restante para la evaluación del modelo.
"""
st.write(texto3)
col1, col2 = st.columns(2)
with col1:
    st.write("**Núm. de señales en el conjunto de entrenamiento**")
    for c in clases:
        st.write("Clase "+c+": "+ str(len([x for x in yEntrenamiento if x==clases.index(c)])) + " señales, etiqueta: " + str(clases.index(c)))
    with st.expander("Ver distribución:" + cadena):
        dibujarDist(yEntrenamiento,clases)
with col2:
    st.write("**Núm. de señales de cada clase para el test**")
    for c in clases:
        st.write("Clase "+c+": "+ str(len([x for x in yTest if x==clases.index(c)])) + " señales, etiqueta: " + str(clases.index(c)))
    with st.expander("Ver distribución:" + cadena):
        dibujarDist(yTest,clases)

st.write("Si considera que el reparto de datos de entrenamiento y test no está equilibrado puede volver a realizar un reparto aleatorio pinchando en el botón de Volver a barajar")

###################################################################
# Construcción del hiperplano de separación
###################################################################

st.subheader("Hiperplano de separación", divider="red")
st.write("Puede variar el parámetro de regularización C, que controla el equilibrio entre el margen del hiperplano y el número de datos clasificados erroneamente.")

cSelect=st.slider("C: ", min_value=1,max_value=100 ,value=10)
vS=st.checkbox("Ver vectores soporte")

modelo = LinearSVC(C = cSelect)
modelo.fit(xEntrenamiento.values, yEntrenamiento.ravel())

dibujarHiperplano(modelo,xEntrenamiento,yEntrenamiento,vS)

###################################################################
# Métricas de evaluación
###################################################################

st.subheader("Métricas de evaluación", divider="red")
st.write("A continuación se muestra la proporción de datos clasificados correctamente (del conjunto de datos reservados para el test) y la matriz de confusión, donde en el eje horizontal se muestra la etiqueta que predice el modelo y en el vertical la etiqueta real.")
matrizConfusion3(modelo,clases,xTest,yTest)