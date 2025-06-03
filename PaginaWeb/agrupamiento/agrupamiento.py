import streamlit as st

#################################
# Aprendizaje automático
#################################

st.title("Aprendizaje automático")

texto1="""
Las técnicas de aprendizaje automático se centran en desarrollar algoritmos para que las máquinas realicen un aprendizaje. Son una herramienta muy útil para extraer conocimiento oculto en los datos. Dependiendo de si las instancias o ejemplos de entrenamiento están o no etiquetados, es decir, si existe información o no de la clase a la que pertenece cada instancia, se pueden agrupar estas técnicas en dos categorías:

*   **Aprendizaje no supervisado**
*   **Aprendizaje supervisado**

Junto con estas dos técnicas de aprendizaje, existe un conjunto de problemas en los que a cada instancia se le asocia cierta información (refuerzo)
de forma retardada en el tiempo. En el aprendizaje por refuerzo se trata de adquirir conocimiento (en base a recompensas y penalizaciones) que determine cuál es 
la próxima acción a realizar dado el estado en el que se encuentra el sistema. Estos algoritmos son muy utilizados en robótica y control
industrial, pero no se van a tratar en esta página. 

En los casos en que existe una clasificación previa de las instancias se habla de aprendizaje automático supervisado. En este tipo de algoritmos se establece una relación entre las 
entradas y las salidas. Se encargan de problemas de clasificación y de regresión. Cuando los datos están sin etiquetar, es el algorimo el encarado de deducir los patrones ocultos en las instancias
buscando similitudes y asociacinones.

Para comenzar se abordarán las técnicas de aprendizaje por agrupación no supervisado, con el objetivo de buscar patrones ocultos en las señales temporales de fusión termonuclear que se están tratando.
"""
st.markdown(texto1)
st.header("Aprendizaje por agrupamiento")

texto2="""
En este apartado se van a analizar dos algoritmos de aprendizaje automático no supervisado por agrupación: **agrupamiento jerárquico** y **k-medias**.

Se aplicarán estos algoritmos a las señales de fusión termonuclear con el objetivo de detectar formas de onda similares. 
Debe tenerse en cuenta que, como datos de entrenamiento de los algoritmos, no se van a utilizar las señales digitales originales, 
sino que es preferible preprocesar previamente los datos utilizando la transformada discreta de Fourier o la transformada 
de Wavelet discreta. Estas técnicas permiten reducir de modo significativo la dimensionalidad de los datos y simplificar el 
análisis.
"""
st.markdown(texto2)