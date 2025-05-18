import streamlit as st
import pandas as np
import pywt
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib as mpl
import plotly.express as px
from PaginaWeb.funciones.fun import *

st.title("SVM para clasificación multinomial")

texto1=r"""
Las SVM se pueden generalizar para la clasificación en más de dos categorías. Se distinguen dos enfoques:
- Métodos indirectos: se basan en descomponer el problema de clasificación múltiple en problemas de clasificación binaria y aplicar a estos problemas los algoritmos de SVM vistos en el apartado anterior.
    - **Uno frente a uno**: se parte de un problema de clasificación en K clases. Dada una clase se construyen k-1 SVMs para comparar esta clase con cada una de las clases restantes  . Es decir, en total se construyen k(k-1)/2 hiperplanos de clasificación. Dada una instancia para clasificarla en una de las k posibles clases se emplean los k(k-1)/2 clasificadores generados y se anota las veces que la instancia se asigna a cada una de las clases. Finalmente, la clase ganadora será la que registre un mayor número de asignaciones.
    
    - **Uno frente al resto**: en este caso se van construir un número menor de hiperplanos de clasificación, en concreto k. En cada SVM se compara una de las clases con las k-1 clases restantes. Para construir el hiperplano de clasificación binario j-esimo se consideran los ejemplos de entrenamientos de la clase j como positivos (+1) y los de las restantes clases como negativos (-1). A la hora de asignar una nueva instancia ($x_{nueva}$) a una de las clases se utilizan los k clasificadores y se asigna a la clase para la cual el valor del clasificador binario $\omega^{t}\Phi(x_{nueva})$ sea mayor

- Métodos directos: no trata de descomponer el proceso de clasificación múltiple en procesos de clasificación binaria sino que realiza un único proceso de optimización combinando los problemas de clasificación binaria en un única función objetivo.
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
    if (getExternos() is not None) and (esPosibleClasBin(getExternos())):
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
    sDefecto=tiposSeries[:3]
    optionTs=st.multiselect(
        "Seleccione un tipo de señal",
        options=tiposSeries,
        default=sDefecto   
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
    if len(seriesSelect):
        st.write("Número de observaciones para cada señal: " + str(len(datosSelect)))

    if(len(optionTs)!=2):
        if(len(optionTs)<2):
            st.write("**Debe elegir al menos dos tipos se series distintos**")
            seriesSelect=[]
        #    st.write(tiposSeries[:2])
            for ts in tiposSeries[:2]:
                for ns in nombreSeries:
                    if ts in ns:
                        seriesSelect.append(ns)
            listaS=['t']
            listaS.extend(seriesSelect)
            datosSelect=datos[listaS]


texto2="""
A continuación se muestra un desplegable para seleccionar las señales a las que se pretende aplicar los algoritmos SVM para clasificación binaria.
Debe seleccionar dos tipos de series distintas.
"""
st.subheader("Preprocesamiento", divider="red")
texto7="""
Se realiza el **preprocesamiento** de las señales mediante transforma wavelet. 
Se fija un nivel de resolución M común para todas las señales. Cada instancia, o ejemplo 
de entrenamiento del algoritmo, estará formada por los coeficientes de aproximación para este nivel de resolución. 
"""
st.write(texto7)

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

st.subheader("Reducción de la dimensionalidad", divider="red")


datosCP=componentesPrincipales2(datosP)
if normalizar:
    datosCP=pd.DataFrame(scale(datosCP),index=seriesSelect,columns=['pc1','pc2'])

texto3="""
Para poder visualizar los vectores soporte y el hiperplano de separación es posible reducir la dimensionalidad de los datos a dos dimensiones. 
Para ello se utiliza el **análisis de componentes principales**."""
st.write(texto3)

texto4=r"""Se parte de un vector $X = (X_{1}, \dots, X_{p})^{t}$ de p dimensiones y se desea pasar a un vector reducido $Z = (X_{1}, \dots ,Z_{r})^{t}$, con $r<p$, obtenido a partir de $X$ y que contenga la máxima información (dispersión) que posee $X$. 
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
    st.write(texto4)

with st.expander("Ver porcentaje de varianza explicada"):
    componentesPrincipales(datosP)

cadena=''
if normalizar:
    cadena=' normalizadas'
with st.expander("Ver primera y segunda componentes principales"+cadena):
    st.write(datosCP)

st.subheader("Ejemplos para el entrenamiento y para el test", divider="red")
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
El siguiente paso es dividir el conjunto de señales en ejemplos de entrenamiento y test. Se tomará el 80% de las instancias para el entrenamiento del modelo y el 20% restante para la evaluación del modelo.
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

st.write("Si considera que el reparto de ejemplos de entrenamiento y test no está equilibrado puede volver a realizar un reparto aleatorio pinchando en el botón de Volver a barajar")

st.subheader("Hiperplano de separación", divider="red")

col1, col2, col3 = st.columns(3)
with col1:
    cSelect=st.slider("C: ", min_value=1,max_value=100 ,value=10)
with col2:
        optionK=st.selectbox(
            "Seleccione el kernel",
            options=['rbf','linear','poly','sigmoid']    
        )    
with col3:
        if optionK=='poly':
            gSelect=st.slider("Grado kernel polinomial: ", min_value=3,max_value=10 ,value=3)
            c0Select=st.slider("Coeficiente independiente: ", min_value=0 ,value=0)
        elif optionK=='sigmoid':
            c0Select=st.slider("Coeficiente independiente: ", min_value=0 ,value=0)
if optionK=='poly':
    modelo=SVC(C=cSelect,kernel=optionK,degree=gSelect,coef0=c0Select)
elif optionK=='sigmoid':
    modelo=SVC(C=cSelect,kernel=optionK,coef0=c0Select)
else:
    modelo=SVC(C=cSelect,kernel=optionK)

modelo.fit(xEntrenamiento.values, yEntrenamiento.ravel())
#modelo.fit(xEntrenamiento.values, yEntrenamiento.ravel())
dibujarHiperplanoMulti(modelo,c,xEntrenamiento,yEntrenamiento,optionK)

st.subheader("Métricas de evaluación", divider="red")
matrizConfusion3(modelo,clases,xTest,yTest)

clases2,x2,y2=etiquetar(datosP)
xEntrenamiento2, xTest2, yEntrenamiento2, yTest2 = train_test_split(x2,y2.values.reshape(-1,1),random_state=1234,train_size=0.8,shuffle=True)

texto6="""Para más de dos dimensiones no podemos visualizar los hiperplanos de separación en el plano bidimensional pero sí se pueden
ajustar los modelos y calcular medidas de ajuste y las matrices de confusión. A continuación, se compara el ajuste realizado para
los coeficientes de aproximación de las señales procesadas mediante la transformada wavelet (con el nivel de profundidad seleccionado más arriba)."""

st.write(texto6)
cSelect2=st.slider(" C:", min_value=1,max_value=100 ,value=10,key=2)
gSelect2=st.slider(" Grado kernel polinomial : ", min_value=3,max_value=10 ,value=3,key=3)
c0Select2=st.slider(" Coeficiente independiente: ", min_value=0 ,value=0,key=4)

col1, col2= st.columns(2)
with col1:
    modelo1=SVC(C=cSelect2,kernel='rbf')
    modelo1.fit(xEntrenamiento2.values, yEntrenamiento2.ravel())
    st.write("**Kernel Gaussiano**")
    matrizConfusion3(modelo1,clases2,xTest2,yTest2)

    modelo2=SVC(C=cSelect2,kernel='poly',degree=gSelect2,coef0=c0Select2)
    modelo2.fit(xEntrenamiento2.values, yEntrenamiento2.ravel())
    st.write("**Kernel polinomial de orden**" + str(gSelect2))
    matrizConfusion3(modelo2,clases2,xTest2,yTest2)    

with col2:
    modelo3=SVC(C=cSelect2,kernel='linear')
    modelo3.fit(xEntrenamiento2.values, yEntrenamiento2.ravel())
    st.write("**Kernel Gaussiano**")
    matrizConfusion3(modelo3,clases2,xTest2,yTest2)

    modelo4=SVC(C=cSelect2,kernel='sigmoid',coef0=c0Select2)
    modelo4.fit(xEntrenamiento2.values, yEntrenamiento2.ravel())
    st.write("**Kernel sigmoidal**")
    matrizConfusion3(modelo4,clases2,xTest2,yTest2)     