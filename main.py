import streamlit as st

pages = {
    "SEÑALES DIGITALES" : [
        st.Page("PaginaWeb/senhales/introduccion.py",title="Introducción", default=True),
        st.Page("PaginaWeb/senhales/senhales.py", title="Señales digitales")
    ],
    "PREPROCESADO DE SEÑALES" : [
        #st.Page("preprocesado/preprocesado.py", title="Preprocesado de señales"),
        st.Page("PaginaWeb/preprocesado/fourier.py", title="Transformada de Fourier"),
        st.Page("PaginaWeb/preprocesado/wavelet.py", title="Wavelets"),
        st.Page("PaginaWeb/preprocesado/ajuste.py", title="Medidas de ajuste"),
        st.Page("PaginaWeb/preprocesado/similitud.py", title="Medidas de similitud")        
    ],
    "APRENDIZAJE POR AGRUPAMIENTO" : [
        st.Page("PaginaWeb/agrupamiento/agrupamiento.py", title="Introducción"),
        st.Page("PaginaWeb/agrupamiento/cluster.py", title="Agrupamiento jerárquico"),
        st.Page("PaginaWeb/agrupamiento/kmedias.py", title="k-medias"),
        st.Page("PaginaWeb/agrupamiento/ncluster.py", title="Número de clústeres")

    ],
    "MÁQUINAS DE VECTOR SOPORTE" : [
        st.Page("PaginaWeb/clasificacion/clasificacion.py",title="Máquinas de vectores soporte (SVM)"),
        st.Page("PaginaWeb/clasificacion/svmBinaria.py",title="Ejemplos separables linealmente"),
        st.Page("PaginaWeb/clasificacion/svmBinariaCs.py",title="Ejemplos cuasi-separables linealmente"),
        st.Page("PaginaWeb/clasificacion/svmBinariaNs.py",title="Ejemplos no separables linealmente"),
        st.Page("PaginaWeb/clasificacion/svmMultinomial.py",title="SVMs para clasificación multinomial")
    ]
}

pg = st.navigation(pages)
pg.run()

