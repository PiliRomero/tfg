import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmath
import math
import pywt
from scipy.cluster.hierarchy import dendrogram 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import matplotlib as mpl
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from scipy.interpolate import CubicSpline
import plotly.express as px

# Carga de señales de fusion termonuclear normalizadas
datosTF=pd.read_csv("./PaginaWeb/datos/sMuestreadasN.csv")

# Se obtiene el nombre de las series
nombreSeriesTF=list(datosTF.columns)[1:]

# Se obtiene el tipo de series
tiposSeriesTF=np.unique(list(map(lambda x: x[0:x.find('_')],nombreSeriesTF)))

if "datosExternos" not in st.session_state:
    st.session_state.datosExternos = None

def setDatosExternos(df):
    st.session_state.datosExternos = df

def getExternos():
    return st.session_state.datosExternos

##############################
# Preprocesar señales
##############################

# Función que dado un dataframe de señales, normaliza las señales y las muestrea a un intervalo regular
# @ param   df      dataframe de señales (cada columna es una señal) que se quier normalizar
# @ param   frec    frecuencia de remuestreo
# @ return  dataframe con las señales normalizadas  y muestreadas
def preprocesarSC(df,frec=None):
    df=df.dropna()
    t=df.columns[0]
    df=df.sort_values(t)
    t0=min(df[t])
    tn=max(df[t])
    if frec==None:
        difTiempo=df[t][1:]-df[t][:-1]
        if max(difTiempo)-min(difTiempo)>1 + 1e-15:
            frec=min(difTiempo)
    dfN=pd.DataFrame()
    if frec != None:
        tiempos=np.arange(t0,tn,frec)
        dfN['t']=tiempos
        for i in df.columns[1:]:
            cs=CubicSpline(df[t],df[i])
            interpolados=cs(tiempos)
            dfN[i]=(interpolados-min(interpolados))/(max(interpolados)-min(interpolados))
    else:
        dfN['t']=df[t]
        for i in df.columns[1:]:
            dfN[i]=(df[i]-min(df[i]))/(max(df[i])-min(df[i]))
    return dfN

# Función que dado un dataframe comprueba si es posible aplicar los algoritmos de agrupamiento
# Se establece un mínimo de 5 señales para aplicar estos algoritmos
# @ param   df  dataframe con las señales de entrada
# @ return  True si es posible aplicar el algoritmo de agrupamiento, False en otro caso   
def esPosibleAgrupamiento(df):
    nS=list(df.columns)[1:]
    if len(nS)<5:
        st.write("Se pide un mínimo de 5 series para apllicar los algoritmos de clasificacion")
        return False
    else:
        return True
    
# Función que dato un data frame comprueba si es posible aplicar el algoritmo SVM binaria
# @ param   df  dataframe con las señales de entrada
# @ return  True si es posible aplicar el algoritmo de calsificación, False en otro caso
def esPosibleClasBin(df):
    nS=list(df.columns)[1:]
    for i in nS:
        if '_' not in i:
            st.write("El nombre de las series no cumplen los criterios fijados: TIPOSERIE_NOMBRESERIE")
            return False
    tS=np.unique(list(map(lambda x: x[0:x.find('_')],nS)))
    if len(tS)<2:
        st.write("Se precisan series de almenos dos dipologías distintas para aplicar algoritmos de clasificación")
        return False
    else:
        return True

# Función que dado el dataframe subido por el usuario llama a la función correspondiente para preprocesar las señales (normalicarlas y muestrearlas)
# En caso de que no se pueda preprocesar (p.ej por tener caracteres no numéricos) devuelve un mensaje de error  
# @ param   df      dataframe con las señales de entrada
# @ param   frec    frecuencia de remuestreo     
# @ return  señales normalizadas y muestreadas  
def getDatosExternosN(df,frec=None):
    try:
        df=df.dropna()
        if len(df)<20:
            st.write("Se requier un mínimo de 20 observaciones para cada serie. Revise el tamaño de las series y los #NA")
            return None
        else:
            return preprocesarSC(df,frec)
    except:
        st.write("El fichero introducido no es correcto. Compruebe que cumple todas las condiciones exigidas y que no tiene caracteres no numéricos")
        return None


################################
# Obtener datos de entrenamiento
################################
@st.cache_data
def getDatosTF():
    return datosTF
@st.cache_data
def getNombreSeriesTF():
    return nombreSeriesTF
@st.cache_data
def getTipoSeriesTF():
    return tiposSeriesTF

################################
# Generales
################################
# Método que dibuja una señal
# @ param   serie   señal que se quiere dibujar. Objeto de la clase Series
# @ param   ventana booleano que indica si se aplica la función ventana hamming (FFT)
def dibujarSerie(serie,ventana=False):
    fig1, ax1 = plt.subplots()
    if ventana==True:
        ax1.plot(serie.index,serie.values * np.hamming(len(serie.index)))
    else:
        ax1.plot(serie)
    ax1.set_xlabel('Tiempo')
    if ventana==True:
        ax1.set_ylabel('Señal*ventana')
        ax1.set_title('Evolución de la señal*ventana '+ serie.name)
    else:
        ax1.set_ylabel('Señal')
        ax1.set_title('Evolución de la señal '+ serie.name)
    st.pyplot(fig1)
    plt.cla()
    plt.close(fig1)
    

#################################
# Transformada de Fourier
#################################

# Función que devuelve la amplitud más grande (en módulo) de la transformada rápida de Fourier
# @ param   serie   señal de entrada. Objeto de la clase Series
# @ param   ventana booleano que indica si se aplica la función ventana hamming (FFT)
# @ return  módulo máximo de la FFT
def moduloMaximo(serie,ventana=False):
    if ventana==True:
        s=serie.values * np.hamming(len(serie.index))
        return max(abs(np.fft.fftshift(np.fft.fft(s))))
    else:
        return max(abs(np.fft.fftshift(np.fft.fft(serie.values))))

# Función que calcula el percentil 99 de la amplitud (en modulo) de la transformada rápida de Fourier
# @ param   serie   señal de entrada. Objeto de la clase Series
# @ param   ventana booleano que indica si se aplica la función ventana hamming (FFT)
# @ return  percentir 99 del módulo de la FFT
def percentil(serie,ventana=False):
    if ventana==True:
        s=serie.values * np.hamming(len(serie.index))
        return np.percentile(abs(np.fft.fftshift(np.fft.fft(s))),99)
    else:
        return np.percentile(abs(np.fft.fftshift(np.fft.fft(serie.values))),99)

# Método que dibuja dos gráficas con la parte real e imaginaria de la transformada rápida de Fourier
# @ param   serie   señal de entrada. Objeto de la clase Series
# @ param   ventana booleano que indica si se aplica la función ventana hamming (FFT)
def dibujarTransformada(serie,ventana=False):
    if ventana==True:
        s = serie.values * np.hamming(len(serie.index))
    else:
        s = serie.values
    
    transformadaFourier=np.fft.fftshift(np.fft.fft(s))
    timestep=serie.index[1]-serie.index[0]
    frecuencia=np.fft.fftshift(np.fft.fftfreq(len(serie),d=timestep))

    fig1=plt.figure(figsize=(10,6))
    f1= fig1.add_subplot(221)
    f1.vlines(frecuencia,[0],transformadaFourier.real)
    f1.set_xlabel('Frecuencia (Hz)')
    f1.set_ylabel('Parte real')
    f1.set_title('Parte real de la FFT: '+serie.name)

    f2=fig1.add_subplot(222)
    f2.vlines(frecuencia,[0],transformadaFourier.imag)
    f2.set_xlabel('Frecuencia (Hz)')
    f2.set_ylabel('Parte imaginaria')
    f2.set_title('Parte imaginaria de la FFT: '+serie.name)
    st.pyplot(fig1)
    plt.cla()
    plt.close(fig1)


# Método que dibuja la fase y el módulo de la transformada rápida de fourier 
# @ param   serie   señal de entrada. Objeto de la clase Series
# @ param   ventana booleano que indica si se aplica la función ventana hamming (FFT)
def dibujarTransformadaMF(serie,ventana=False):
    if ventana==True:
        s = serie.values * np.hamming(len(serie.index))
    else:
        s = serie.values

    transformadaFourier=np.fft.fftshift(np.fft.fft(s))
    timestep=serie.index[1]-serie.index[0]
    frecuencia=np.fft.fftshift(np.fft.fftfreq(len(serie),d=timestep))
    fases = list(map(cmath.phase,transformadaFourier))

    fig1=plt.figure(figsize=(10,6))
    f1= fig1.add_subplot(221)
    f1.vlines(frecuencia,0,abs(transformadaFourier))
    f1.set_xlabel('Frecuencia (Hz)')
    f1.set_ylabel('Modulo')
    f1.set_title('Módulo de la FFT: '+serie.name)

    f2=fig1.add_subplot(222)
    f2.vlines(frecuencia,0,fases)
    f2.set_xlabel('Frecuencia (Hz)')
    f2.set_ylabel('Fase')
    f2.set_title('Fase de la FFT: '+serie.name)
    st.pyplot(fig1)
    plt.cla()
    plt.close(fig1)

# Método que dibuja la transformada inversa de Fourier tras anular los coeficentes cuyo módulo está por debajo
# de cierto umbral (módulo). También es posible pasar el número de coeficientes que se desan que sean no nulos
# En este caso el programa considerará no nulos los de mayor módulo
# @ param   serie   señal de entrada. Objeto de la clase Series
# @ param   ventana booleano que indica si se aplica la función ventana hamming (FFT)
def dibujarSerieInv(serie, modulo=None, numero=None, ventana=False):
    if ventana==True:
        s = serie.values * np.hamming(len(serie.index))
    else:
        s = serie.values
    transformadaFourier=np.fft.fftshift(np.fft.fft(s))
    transformadaFourierSR=[]
    if modulo!=None:
        for t in transformadaFourier:
            if(abs(t)>=modulo):
                transformadaFourierSR.append(t)
            else:
                transformadaFourierSR.append(0+0j)
    else:
        modulos=abs(transformadaFourier)
        modulos.sort()
        m=modulos[::-1][numero-1]
        for t in transformadaFourier:
            if(abs(t)>=m):
                transformadaFourierSR.append(t)
            else:
                transformadaFourierSR.append(0+0j)
    yInversaSR = np.fft.ifft(np.fft.ifftshift(transformadaFourierSR))

    if ventana==True:
        yInversaSR=yInversaSR/np.hamming(len(serie.index))

    fig1, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(12,4))
    ax1.plot(serie)
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Serie original')
    ax1.set_title('Serie original ' + serie.name)

    ax2.plot(serie.index,yInversaSR.real, label='real')
    ax2.plot(serie.index,yInversaSR.imag, label='imaginaria')
    ax2.legend()
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Serie reconstruida')
    ax2.set_title('FFT inversa ' + serie.name)
    st.pyplot(fig1)
    plt.cla()
    plt.close(fig1)

#########################
# Wavelets
##########################

# Método que dibuja los coeficientes de aproximación y detalle de la transormada wavelet de un sólo nivel para una señal
# y una wavelet madre pasadas como parámetros
# @ param   serie   señal de entrada. Objeto de la clase Series
# @ param   w       wavelet madre
def digujarCoeficientes(serie,w):
    cA,cD = pywt.dwt(serie.values,w)
    paso=2*(serie.index[1]-serie.index[0])
    m=serie.index[0]+np.arange(0, len(cA))*paso
    fig1=plt.figure(figsize=(12,6))
    f1= fig1.add_subplot(221)
    f1.plot(m,cA)
    f1.set_ylabel('$c$')
    f1.set_title('Coeficientes de aproximación: '+ serie.name)

    f2= fig1.add_subplot(222)
    f2.vlines(m,0,cD)
    f2.set_ylabel('$d$')
    f2.set_title('Coeficientes de detalle: '+ serie.name)
    st.pyplot(fig1)
    plt.cla()
    plt.close(fig1)

# Método que dibuja los coeficientes de aproximación de una wavelet dado un nivel de despomposición m y una
# wavelet madre m y la transformada Wavelet inversa una vez despreciados los coeficientes de detalle
# @ param   serie   señal de entrada. Objeto de la clase Series
# @ param   w       wavelet madre
# @ param   m       nivel de descomposición
def dibujarCAeInv(serie,w,m):
    coeficientes = pywt.wavedec(serie.values,w, level=m)
    for i in range(1, m):
        coeficientes[-i]=np.zeros_like(coeficientes[-i])
    ywR=pywt.waverec(coeficientes,w)
    paso=2**m*(serie.index[1]-serie.index[0])
    mm=serie.index[0]+np.arange(0, len(coeficientes[0]))*paso
    
    fig1=plt.figure(figsize=(10,6))
    f1= fig1.add_subplot(221)
    f1.plot(mm,coeficientes[0])
    f1.set_xlabel('Tiempo')
    f1.set_ylabel('ci')
    f1.set_title('Coeficientes de aproximación: '+serie.name)

    f2= fig1.add_subplot(222)
    f2.plot(serie.index,ywR[:len(serie.index)])
    f2.set_xlabel('Tiempo')
    f2.set_ylabel('IDWT')
    f2.set_title('IDWT: '+serie.name)
    st.pyplot(fig1)
    plt.cla()
    plt.close(fig1)
    return len(coeficientes[0])

# Método que dibuja la gráfica de una wavelet madre discreta pasada como parámetro
# @ param   w       wavelet madre
def dibujarWaveletDiscreta(w):
    [y,x]=pywt.DiscreteContinuousWavelet(w).wavefun()[1:]
    fig1,ax1=plt.subplots()
    ax1.plot(x,y)
    ax1.set_title(w)
    st.pyplot(fig1)
    plt.cla()
    plt.close(fig1)

# Método que dibuja la gráfica de una wavelet madre continua pasada como parámetro
# @ param   w       wavelet madre
def dibujarWaveletContinua(w):
    if w in ['cmor','shan']:
        w +='1-1'
    elif w=='fbsp':
        w +='1-1.5-1.0'
    fig1,ax1=plt.subplots()
    [psi,x]=pywt.ContinuousWavelet(w).wavefun(10)
    ax1.plot(x,np.real(psi),label="real")
    ax1.plot(x,np.imag(psi),label="imag")
    ax1.set_title(w)
    st.pyplot(fig1)
    plt.cla()
    plt.close(fig1)

# Método que dibuja la gráfica de una wavelet madre
# @ param   w       wavelet madre
def dibujarWavelet(w):
    wavelistD=pywt.wavelist(kind="continuous")
    wavelistC=pywt.wavelist(kind="discrete")
    if w in wavelistD:
        dibujarWaveletContinua(w)
    else:
        dibujarWaveletDiscreta(w)
    

#############################
# Medidas de ajuste
#############################

# Método que calcula y dibuja el ECM y el EAM entre la serie original y la serie suavizada tras aplicarle la transformada wavelet
# @ param   serie   señal de entrada. Objeto de la clase Series
# @ param   w       wavelet madre
# @ param   nmax    nivel máximo de descomposición
def dibujarECM(serie,w,nmax):
    ecmF=[]
    ecmW=[]
    eamF=[]
    eamW=[]
    s = serie.values * np.hamming(len(serie.index))
    transformadaFourier=np.fft.fftshift(np.fft.fft(s))
    modulos=abs(transformadaFourier)
    modulos.sort()
#    for n in range(10,200,10):
    for n in np.arange(2,nmax,1):
        transformadaFourierSR=[]  
        m=modulos[::-1][n-1]
        for t in transformadaFourier:
            if(abs(t)>=m):
                transformadaFourierSR.append(t)
            else:
                transformadaFourierSR.append(0+0j)
        yInversaSR = np.fft.ifft(np.fft.ifftshift(transformadaFourierSR))
        yTF=yInversaSR.real/np.hamming(len(serie.index))
        u=serie.values-yTF[:len(serie.index)]
        e1=1/len(serie.index)*np.sum(u**2)
        e2=1/len(serie.index)*np.sum(abs(u))
        ecmF.append(e1)
        eamF.append(e2)
    ejex1=np.arange(2,nmax,1)

    for m in range(1, math.trunc(math.log2(len(serie.index)))-1):
        coeficientes = pywt.wavedec(serie.values,w, level=m)
        for i in range(1, m):
            coeficientes[-i]=np.zeros_like(coeficientes[-i])
        ywR=pywt.waverec(coeficientes,w)
        u=serie.values-ywR[:len(serie.index)]
        e1=1/len(serie.index)*np.sum(u**2)
        e2=1/len(serie.index)*np.sum(abs(u))
        ecmW.append(e1)
        eamW.append(e2)
    ejex2=np.arange(1,math.trunc(math.log2(len(serie.index)))-1)

    fig1=plt.figure(figsize=(12,10))
    f1= fig1.add_subplot(221)
    f1.plot(ejex1,ecmF)
    f1.set_xlabel('N')
    f1.set_ylabel('ECM')
    f1.set_title('ECM: '+ serie.name)

    f2= fig1.add_subplot(222)
    f2.plot(ejex2,ecmW)
    f2.set_xlabel('M')
    f2.set_ylabel('ECM')
    f2.set_title('ECM: '+serie.name)

    f3=fig1.add_subplot(223)
    f3.plot(ejex1,eamF)
    f3.set_xlabel('N')
    f3.set_ylabel('EAM')
    f3.set_title('EAM: '+ serie.name)

    f4=fig1.add_subplot(224)
    f4.plot(ejex2,eamW)
    f4.set_xlabel('M')
    f4.set_ylabel('EAM')
    f4.set_title('EAM: '+ serie.name)

    st.pyplot(fig1)
    plt.cla()
    plt.close(fig1)


##################################
# Clustering
##################################

# Función que dado un data frame de datos, una lista con el nombre de las series seleccionadas,
# un nivel de descomposición y una wavelet madre preprocesa las series mediante la transformada Wavelet
# @ param   datos   dataframe con conjunto de datos con todas las señales
# @ param   series  nombre de las señales seleccionadas
# @ param   m       nivel de descomposición
# @ param   w       wavelet madre
# @ return  lista donde cada elemento corresponde con una señal preprocesada mediante la transformada wavelet
def obtenerSeriesPreprocesadas(datos,series,m,w):
    lista=[]
    for s in series:
        s=pd.Series(datos[s])
        s.index=datos['t']
        lista.append(s)
    listaP=[]
    for s in lista:
        c=pywt.wavedec(s,wavelet=w,level=m)
        listaP.append(c[0])
    return listaP

# Menu para seleccionar la medida de enlace en el clústering
# @ return nombre de la medida de enlace seleccionada entre las opciones ward, complete, average y single
def linkage():
    l=['ward', 'complete', 'average', 'single']
    
    col1, col2 = st.columns(2)
    with col1:
        option=st.selectbox(
            "Linkage",
            options=l  
    )
    with col2:
        if option=='ward':
            texto="""Minimiza la magnitud en la que se incrementa la suma de cuadrados de los errores cuando se unen dos clases."""
        elif option=='complete':
            texto="""El encadenamiento completo toma la distancia entre los dos puntos más alejados de las clases."""
        elif option=='single':
            texto="""El encadenamiento simple toma como distancia entre dos clases la distancia ente los dos puntos más próximos de las clases"""
        elif option=='average':
            texto="""Utiliza el promedio de las distancias de cada observación de las dos clases"""
        st.write(texto)
    return option

# Menú para seleccionar la distancia
# @ return distancia seleccionada entre las opciones: euclidean, manhattan, cosine
def medida(link): 
    if link=='ward':
        medidas=['euclidean']
    else:
        medidas=['euclidean', 'manhattan', 'cosine']
    col1, col2 = st.columns(2)
    with col1:
        option=st.selectbox(
            "Metric",
            options=medidas      
    )
    with col2:
        if option=='euclidean':
            texto=r"""
            $$
            d(u,v) = \sqrt{\sum_{i}(u_{i}-v_{i})^{2}}
            $$
            """
        elif option=='manhattan':
            texto=r"""
            $$
            d(u,v)= \sum_{i} \left| u_{i} - v_{i} \right|
            $$
            """
        elif option=='cosine':
            texto=r"""
            $$
            d(u,v)=\frac{\left| u \cdot v \right|}{\| u\| \|v \|}
            $$
            """
        st.write(texto)
    return option

# Método para dibujar un dendograma que cambie el color de los clústeres en función de la altura de la línea de corte
# @ param   model   modelo de agrupamiento ajustado
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)

# Método que llama a la encargada de dibujar el dendograma pasándole los parámetros oportunos y dibuja la línea de corte
# @ param   model           modelo de agrupamiento ajustado
# @ param   alturaCorte     altura de corte del dendograma
# @ param   enlace          medida de enlace
# @ param   etiquetas       nombres de las señales
def dibujarDendograma(modelo,alturaCorte,medida,enlace,etiquetas):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_dendrogram(modelo, color_threshold=alturaCorte, ax=ax, labels=etiquetas, leaf_rotation=90)
    ax.set_title("Distancia "+str(medida) +" Linkage "+str(enlace))
    ax.axhline(y=alturaCorte, c = 'black', linestyle='--', label='altura corte')
    ax.legend()
    st.pyplot(fig)
    plt.cla()
    plt.close(fig)

def dibujarDendograma2(modelo,alturaCorte,medida,enlace):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_dendrogram(modelo, color_threshold=alturaCorte, ax=ax)
    ax.set_title("Distancia "+str(medida) +" Linkage "+str(enlace))
    ax.axhline(y=alturaCorte, c = 'black', linestyle='--', label='altura corte')
    ax.legend()
    st.pyplot(fig)
    plt.cla()
    plt.close(fig)

# Método que dibuja las gráficas de las señales (coeficientes de aproximación) agrupadas en función del
# clúster al que pertenezca
# @ param   ncluster    número de clústeres
# @ param   datos       dataframe con las señales preprocesadas a dibujar
# @ param   clases      array conel grupo al que pertenece cada señal    
def dibujarClusters(nCluster,datos,clases):
    filas=nCluster//2+nCluster%2    
    col1, col2 = st.columns(2)
    with col1:
        for i in range(0,nCluster,2):
            fig1, ax1 = plt.subplots()
            for k in range(len(datos.to_numpy()[clases==i,:])):
                ax1.plot(datos.to_numpy()[clases==i,:][k,:])
            ax1.set_title('clase '+ str(i))
            st.pyplot(fig1)
            plt.cla()
            plt.close(fig1)
    with col2:
        for i in range(1,nCluster,2):
            fig1, ax1 = plt.subplots()
            for k in range(len(datos.to_numpy()[clases==i,:])):
                ax1.plot(datos.to_numpy()[clases==i,:][k,:])
            ax1.set_title('clase '+ str(i))
            st.pyplot(fig1) 
            plt.cla()
            plt.close(fig1)      

# Método que dibuja un gráfico de barras con el porcentaje de variabilidad explicada por cada componente principal
# y calcula el procentaje de variabilidad explicada por las dos primeras componentes principales
# @ param   datos   dataframe con el conjunto de señales preprocesadas
def componentesPrincipales(datos):
    pca=PCA() 
    datosPCA=pca.fit_transform(datos)
    per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
    varianzaE=np.round((pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])*100,1)
    st.write("El porcentaje de variabilidad explicada por las 2 primeras componentes principales es: ")
    st.write(f'{varianzaE/100:.2%}')
    labels = [str(x) for x in range(1,len(per_var)+1)]
    fig1, ax1 = plt.subplots()
    ax1.bar(x=range(1,len(per_var)+1),height=per_var)
    ax1.tick_params(axis='x', which='both',bottom=False, top=False, labelbottom=False)
    ax1.set_ylabel("Porcentaje de varianza explicada")
    ax1.set_xlabel('Componente principal')
    st.pyplot(fig1)
    plt.cla()
    plt.close(fig1)

# Función que representa en el plano un conjunto de datos tras reducir la dimensionalidad aplicando componentes principales
# @ param   datos   dataframe con el conjunto de señales preprocesadas
def componentesPrincipales2(datos):
    pca=PCA() 
    datosPCA=pca.fit_transform(datos)
    pc1=datosPCA[:,0]
    pc2=datosPCA[:,1]
    #datosCP= np.column_stack((datos.index,pc1,pc2))
    datosCP=pd.DataFrame()
    datosCP['pc1']=pc1
    datosCP['pc2']=pc2
    datosCP['serie']=datos.index
    datosCP.set_index('serie',inplace=True)
    return datosCP

# Función que dibuja en el plano una nube de puntos donde el color varía en función del clúster al que tertenece
# @ param   datos       dataframe con el conjunto de señales preprocesadas
# @ param   clases      array con los grupos a que pertenece cada señal del conjunto de datos
# @ param   nCluster    número de clústeres
def dibujarCluster2d(datos,clases,nCluster):
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(nCluster):
      etiqueta='clase '+ str(k)
      ax.scatter(datos.to_numpy()[clases==k,0],datos.to_numpy()[clases==k,1],label=etiqueta)
    ax.set_title("Gráfico dispersión de las componentes principales")
    ax.set_xlabel('1ª componente principal')
    ax.set_ylabel('2ª componente principal')
    ax.legend()
    st.pyplot(fig)
    plt.cla()
    plt.close(fig)

# Función que dibuja en el plano una nube de puntos donde el color varía en función del clúster al que tertenece
# En este caso el gráfico es iterativo
# @ param   datos       dataframe con el conjunto de señales preprocesadas
# @ param   clases      array con los grupos a que pertenece cada señal del conjunto de datos
# @ param   nCluster    número de clústeres
def dibujarCluster2d_bis(datos,clases,nCluster):
    aux=datos.copy()
    aux['clases']=clases
    aux['clases']=aux['clases'].astype(str)
    aux['serie']=aux.index
    fig = px.scatter(aux, x = "pc1", y = "pc2", color = "clases",hover_name="serie",title="Gráfico dispersión de las componentes principales")   
    fig.update_layout(xaxis_title="Primera componente principal", yaxis_title="Segunda componente principal")
    st.plotly_chart(fig,on_select="rerun")

# Función que dibuja un gráfico para poder determinar el número de clústeres óptimo mediante el método del codo
# @ param   datos   dataframe con el conjunto de señales preprocesadas
def metodoCodo(datos):
    ssd=[]
    for i in range(12):
        kmedias=KMeans(n_clusters=i+1, init='random')
        kmedias.fit_predict(datos)
        ssd.append(kmedias.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1,13),ssd)
    ax.set_title("Método del codo")
    ax.set_xlabel('Número de clústeres')
    ax.set_ylabel('Suma de distancias al cuadrado')
    st.pyplot(fig)
    plt.cla()
    plt.close(fig)

# Función que devuelve el número óptimo de clústeres aplicando el método silhouette
# @ param   datos       dataframe con el conjunto de señales preprocesadas
def optimoSilhouette(datos):
    coef=[]
    for k in range(2,len(datos)):
        kmedias=KMeans(n_clusters=k,init='random')
        clases=kmedias.fit_predict(datos)
        c=silhouette_score(datos,clases)
        coef.append(c)
    kOptimo=coef.index(max(coef))+2
    return kOptimo

###############################
# Algorimos de clasificación
###############################

# Función que etiqueta el conjunto de señales
# @ param   datos       dataframe con el conjunto de señales preprocesadas
# @ retun   clases      nombre de las distintas clases de señales
# @ return  y           etiquetas numeradas 0, 1, 2, ....           
def etiquetar(datos):
    etiquetas=list(map(lambda x: x[0:x.find('_')],datos.index))
    clases=list(np.unique_values(etiquetas))
    datosC=datos.copy()
    datosC['e']=datosC.index
    datosC['y']=datosC['e'].apply(lambda x: clases.index(x[0:x.find('_')])) 
    x=datosC.iloc[:, :-2]
    y=datosC['y']
    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    return clases,x,y

# Función que dibuja el gráfico de sectores con la sitribución del número de señales de cada clase
# @ param   y       código numérico de la clase
# @ param   clases  lista de clases
def dibujarDist(y,clases):
    fig, ax = plt.subplots()
    valores=[]
    cmap=plt.get_cmap("tab20c")
    colores=cmap(np.array([1,2,5,6,9,10]))
    for c in clases:
        valores.append(len([i for i in y if i==clases.index(c)]))
    ax.pie(valores,labels=clases,colors=colores)
    st.pyplot(fig)
    plt.cla()
    plt.close(fig)

# Método que dibuja un hiperplano de separación para datos (bidimensionales) cuasi separables linealmente
# @ param   modelo  modelo SVM lineal ajustado   
# @ param   xE      datos de entrenammiento
# @ param   yE      etiquetas numéricas de las clases a las que pertenecen los datos de entrenamiento
# @ param   vS      True o False indicar si se señalan o no los vectores soporte
def dibujarHiperplano(modelo,xE,yE,vS=False):
    xx = np.linspace(np.min(xE.iloc[:,0]), np.max(xE.iloc[:,0]), 50)
    yy = np.linspace(np.min(xE.iloc[:,1]), np.max(xE.iloc[:,1]), 50)
    Y, X = np.meshgrid(yy, xx)
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    pred_grid = modelo.predict(grid)

    funcionDecision = modelo.decision_function(xE.values)
    indicesVectoresSoporte = np.where(np.abs(funcionDecision) <= 1 + 1e-15)[0]
    vectoresSoporte = xE.iloc[indicesVectoresSoporte]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(grid[:,0], grid[:,1], c=pred_grid, cmap=mpl.colormaps['plasma'],alpha=0.1)
    ax.scatter(xE.iloc[:,0], xE.iloc[:,1], c=yE,cmap=mpl.colormaps['plasma'])
    if vS:
        ax.scatter(vectoresSoporte.iloc[:, 0], vectoresSoporte.iloc[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')

    ax.contour(
        X,
        Y,
        modelo.decision_function(grid).reshape(X.shape),
        colors = 'k',
        levels = [-1, 0, 1],
        alpha  = 0.5,
        linestyles = ['--', '-', '--']
    )
    ax.set_title("Resultados clasificación SVM lineal")
    st.pyplot(fig)
    plt.cla()
    plt.close(fig)

# Método que dibuja la matriz de confusión
# @ param   modelo  modelo ajustado   
# @ param   clases  lista con las clases distintas de los ejemplos de entrenamiento
# @ param   xT      datos para el test
# @ param   yT      etiqueta real de los datos para el test
def matrizConfusion3(modelo,clases,xT,yT):
    predicciones=modelo.predict(xT.values)
    cm=confusion_matrix(yT,predicciones,labels=modelo.classes_)
    #st.write(cm)
    df=pd.DataFrame(cm,index=clases,columns=clases)
    fig=px.imshow(df,labels=dict(x="Predicted label",y="True label"))
    fig.update_layout(dict(title=dict(text="Matriz de confusión")))
    st.write("**Accuracy**: "+str(round(accuracy_score(yT,predicciones),3)))
    st.plotly_chart(fig,on_select="rerun")

# Método que dibuja un hiperplano de separación para datos no separables linealmente
# @ param   modelo  modelo SVM  ajustado   
# @ param   xE      datos de entrenammiento
# @ param   yE      etiquetas numéricas de las clases a las que pertenecen los datos de entrenamiento
# @ param   ker     función kernel para la SVM danos no separables linealmente
# @ param   vS      True o False indicar si se señalan o no los vectores soporte
def dibujarHiperplanoNL(modelo,clases,xE,yE,ker,vS=False):
    xx = np.linspace(np.min(xE.iloc[:,0]), np.max(xE.iloc[:,0]), 50)
    yy = np.linspace(np.min(xE.iloc[:,1]), np.max(xE.iloc[:,1]), 50)
    Y, X = np.meshgrid(yy, xx)
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    pred_grid = modelo.predict(grid)
    Z = modelo.decision_function(grid).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(grid[:,0], grid[:,1], c=pred_grid, cmap=mpl.colormaps['plasma'],alpha=0.1)
    ax.scatter(xE.iloc[:,0], xE.iloc[:,1], c=yE,cmap=mpl.colormaps['plasma'])
    ax.contour(X, Y, Z, colors = 'k', levels = [0], alpha  = 0.5, linestyles = '-')
    ax.set_title('Kernel '+ ker)
    if vS:
        ax.scatter(modelo.support_vectors_[:, 0],modelo.support_vectors_[:, 1],s=200, linewidth=1,facecolors='none', edgecolors='black')
    st.pyplot(fig)
    plt.cla()
    plt.close(fig)

# Función que dibuja un hiperplano de separación multinomial 
# @ param   modelo  modelo SVM  ajustado   
# @ param   xE      datos de entrenammiento
# @ param   yE      etiquetas numéricas de las clases a las que pertenecen los datos de entrenamiento
# @ param   ker     función kernel para la SVM danos no separables linealmente
def dibujarHiperplanoMulti(modelo,clases,xE,yE,ker):
    xx = np.linspace(np.min(xE.iloc[:,0]), np.max(xE.iloc[:,0]), 50)
    yy = np.linspace(np.min(xE.iloc[:,1]), np.max(xE.iloc[:,1]), 50)
    Y, X = np.meshgrid(yy, xx)
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    pred_grid = modelo.predict(grid)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(grid[:,0], grid[:,1], c=pred_grid, cmap=mpl.colormaps['plasma'],alpha=0.1)
    ax.scatter(xE.iloc[:,0], xE.iloc[:,1], c=yE,cmap=mpl.colormaps['plasma'])
    ax.set_title('Kernel '+ ker)
    st.pyplot(fig)
    plt.cla()
    plt.close(fig)

# Método para imprimir la teoría de componentes principales
def imprimirTextoComponentesPrincipales():
    texto=r"""
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
    st.write(texto)