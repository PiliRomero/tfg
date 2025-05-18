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

datosTF=pd.read_csv("./datos/sMuestreadasN.csv")

nombreSeriesTF=list(datosTF.columns)[1:]
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
    
def esPosibleAgrupamiento(df):
    nS=list(df.columns)[1:]
    if len(nS)<5:
        st.write("Se pide un mínimo de 5 series para apllicar los algoritmos de clasificacion")
        return False
    else:
        return True

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

def getDatosExternosN2(df,frec=None):
    df=df.dropna()
    if len(df)<20:
        st.write("Se requier un mínimo de 20 observaciones para cada serie. Revise el tamaño de las series y los #NA")
        return None
    else:
        return preprocesarSC(df,frec)
        
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

def getDatosTF():
    return datosTF

def getNombreSeriesTF():
    return nombreSeriesTF

def getTipoSeriesTF():
    return tiposSeriesTF

################################
# Generales
################################
# Dibujar señal
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
    return fig1


#################################
# Transformada de Fourier
#################################

# Calcular transformada Fourier
def moduloMaximo(serie,ventana=False):
    if ventana==True:
        s=serie.values * np.hamming(len(serie.index))
        return max(abs(np.fft.fftshift(np.fft.fft(s))))
    else:
        return max(abs(np.fft.fftshift(np.fft.fft(serie.values))))

def percentil(serie,ventana=False):
    if ventana==True:
        s=serie.values * np.hamming(len(serie.index))
        return np.percentile(abs(np.fft.fftshift(np.fft.fft(s))),99)
    else:
        return np.percentile(abs(np.fft.fftshift(np.fft.fft(serie.values))),99)

# Dibujar transformada
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
    plt.tight_layout()
    return fig1

# Dibujar transformada modulo fase
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
    return fig1

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
    return fig1

#########################
# Wavelets
##########################

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
    return fig1

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
    return len(coeficientes[0]),fig1

def dibujarWaveletDiscreta(w):
    [y,x]=pywt.DiscreteContinuousWavelet(w).wavefun()[1:]
    fig1,ax1=plt.subplots()
    ax1.plot(x,y)
    ax1.set_title(w)
    return fig1

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
    return fig1

def dibujarWavelet(w):
    wavelistD=pywt.wavelist(kind="continuous")
    wavelistC=pywt.wavelist(kind="discrete")
    if w in wavelistD:
        return dibujarWaveletContinua(w)
    else:
        return dibujarWaveletDiscreta(w)
    

#############################
# Ajuste
#############################

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

    return fig1


##################################
# Clustering
##################################

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

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)

def dibujarDendograma(modelo,alturaCorte,medida,enlace):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_dendrogram(modelo, color_threshold=alturaCorte, ax=ax)
    ax.set_title("Distancia "+str(medida) +" Linkage "+str(enlace))
    ax.axhline(y=alturaCorte, c = 'black', linestyle='--', label='altura corte')
    ax.legend()
    return fig


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
    with col2:
        for i in range(1,nCluster,2):
            fig1, ax1 = plt.subplots()
            for k in range(len(datos.to_numpy()[clases==i,:])):
                ax1.plot(datos.to_numpy()[clases==i,:][k,:])
            ax1.set_title('clase '+ str(i))
            st.pyplot(fig1)       

def componentesPrincipales(datos):
    pca=PCA() 
    datosPCA=pca.fit_transform(datos)
    per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
    varianzaE=np.round((pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])*100,1)
    st.write("El porcentaje de variabilidad explicada por las 2 primeras componentes principales es: "+ str(varianzaE)+"\%")
    labels = [str(x) for x in range(1,len(per_var)+1)]
    fig1, ax1 = plt.subplots()
    ax1.bar(x=range(1,len(per_var)+1),height=per_var)
    ax1.tick_params(axis='x', which='both',bottom=False, top=False, labelbottom=False)
    ax1.set_ylabel("Porcentaje de varianza explicada")
    ax1.set_xlabel('Componente principal')
    st.pyplot(fig1)

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
    return fig

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
# Clasificacion
###############################

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

def dibujarDist(y,clases):
    fig, ax = plt.subplots()
    valores=[]
    cmap=plt.get_cmap("tab20c")
    colores=cmap(np.array([1,2,5,6,9,10]))
    for c in clases:
        valores.append(len([i for i in y if i==clases.index(c)]))
    ax.pie(valores,labels=clases,colors=colores)
    return fig

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
    return fig

def matrizConfusion3(modelo,clases,xT,yT):
    predicciones=modelo.predict(xT.values)
    cm=confusion_matrix(yT,predicciones,labels=modelo.classes_)
    #st.write(cm)
    df=pd.DataFrame(cm,index=clases,columns=clases)
    fig=px.imshow(df,labels=dict(x="Predicted label",y="True label"))
    fig.update_layout(dict(title=dict(text="Matriz de confusión")))
    st.write("**Accuracy**: "+str(round(accuracy_score(yT,predicciones),3)))
    st.plotly_chart(fig,on_select="rerun")

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
    return fig

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
    return fig
