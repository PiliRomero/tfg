import streamlit as st
 

pages = {
    "SEÑALES DIGITALES" : [
        st.Page("senhales/introduccion.py",title="Introducción", default=True),
        st.Page("senhales/senhales.py", title="Señales digitales")
    ],
    "PREPROCESADO DE SEÑALES" : [
        #st.Page("preprocesado/preprocesado.py", title="Preprocesado de señales"),
        st.Page("preprocesado/fourier.py", title="Transformada de Fourier"),
        st.Page("preprocesado/wavelet.py", title="Wavelets"),
        st.Page("preprocesado/ajuste.py", title="Medidas de ajuste"),
        st.Page("preprocesado/similitud.py", title="Medidas de similitud")        
    ],
    "APRENDIZAJE POR AGRUPAMIENTO" : [
        st.Page("agrupamiento/agrupamiento.py", title="Introducción"),
        st.Page("agrupamiento/cluster.py", title="Agrupamiento jerárquico"),
        st.Page("agrupamiento/kmedias.py", title="k-medias"),
        st.Page("agrupamiento/ncluster.py", title="Número de clústeres")

    ],
    "MÁQUINAS DE VECTOR SOPORTE" : [
        st.Page("clasificacion/clasificacion.py",title="Máquinas de vectores soporte (SVM)"),
        st.Page("clasificacion/svmBinaria.py",title="Ejemplos separables linealmente"),
        st.Page("clasificacion/svmBinariaCs.py",title="Ejemplos cuasi-separables linealmente"),
        st.Page("clasificacion/svmBinariaNs.py",title="Ejemplos no separables linealmente"),
        st.Page("clasificacion/svmMultinomial.py",title="SVMs para clasificación multinomial")
    ]
}

pg = st.navigation(pages)
pg.run()