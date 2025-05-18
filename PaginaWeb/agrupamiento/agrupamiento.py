import streamlit as st
st.title("Aprendizaje automático")

texto1="""
Las técnicas de aprendizaje automático se centran en desarrollar algoritmos para que las máquinas realicen un aprendizaje. Son una herramienta muy útil para extraer conocimiento oculto en los datos. Dependiendo de si las instancias o ejemplos de entrenamiento están o no etiquetados, es decir, si existe información o no de la clase a la que pertenece cada instancia, se pueden agrupar estas técnicas en dos categorías:

*   **Aprendizaje no supervisado**
*   **Aprendizaje supervisado**

Para comenzar se abordarán las técnicas de aprendizaje por agrupación no supervisado, con el objetivo de buscar patrones ocultos en las señales temporales de fusión termonuclear que se están tratando.
"""
st.markdown(texto1)
st.header("Aprendizaje por agrupamiento")

texto2="""
En este apartado se van a analizar dos algoritmos de aprendizaje automático no supervisado por agrupación: **agrupamiento jerárquico** y **k-medias**.

Se aplicarán estos algoritmos a las señales de fusión termonuclear con el objetivo de detectar formas de onda similares. 
Debe tenerse en cuenta que, como datos de entrenamiento de los algoritmos, no se van a utilizar las señales digitales originales, 
sino que es preferible preprocesar previamente los datos utilizando la transformada discreta de Fourier o la transformada 
de wavelet discreta. Estas técnicas permiten reducir de modo significativo la dimensionalidad de los datos y simplificar el 
análisis.
"""
st.markdown(texto2)