import streamlit as st

###########################
# PORTADA DE LA PAGINA WEB
###########################

st.image("./PaginaWeb/imagenes/portada.jpg",use_container_width=False)
#st.markdown("<h1 style='text-align: center; color: red;'>Web dinámica para el análisis de series temporales utilizando aprendizaje automático</h1>", unsafe_allow_html=True)
# st.title("Web dinámica para el análisis de series temporales utilizando aprendizaje automático")

left, middle, right = st.columns(3)
if middle.button("Antes de empezar", icon=":material/arrow_drop_down:", use_container_width=True):
    st.write("En esta página web  se abordan las principales técnicas de procesamiento de datos y aprendizaje automático que permiten reducir la disensionalidad de los datos, selección de las principales características y reconocimiento de patrones para datos masivos. Para ilustrar las explicaciones teóricas de las principales técnicas empleadas en el tratamiento de datos se utilizará una base de datos con la evolución temporal de varias señales digitales obtenidas del dispositivo experimental de **fusión termonuclear TJ_II**, aunque también se permite que el usuario pueda utilizar su propio conjunto de datos.")

    st.write("La fusión nuclear es una reacción nuclear en la que varios núcleos de átomos ligeros se unen para formar otro núcleo más pesado. Para conseguir la energía necesaria para que estos núcleos ligeros se aproximen lo suficiente de modo que la atracción de los núcleos supere a las fuerzas de repulsión electrostática se pueden calentar los átomos a altas temperaturas hasta lograr un plasma que estará compuesto por electrones libres y átomos altamente ionizados. Este plasma deberá estar confinado el tiempo suficiente para que se produzca la reacción, por ejemplo mediante la acción de un campo magnético.")

    st.write("El objetivo del proyecto TJ-II es el estudio del comportamiento del plasma confinado magnéticamente.")

