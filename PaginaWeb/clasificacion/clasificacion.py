import streamlit as st
st.title("Máquinas de Vectores Soporte (SVM)")

texto1="""
Las máquinas de vectores soporte (SVM) son algoritmos de aprendizaje automático supervisado utilizados principalmente para resolver problemas de clasificación (binaria y multiclasificación) y regresión.

En esta página se utilizarán las SVM en primer lugar para la clasificación binaria de señales digitales y, posteriormente, se ampliará el análisis para la clasificación en más de dos grupos.

Para evaluar el rendimiento del modelo de la clasificación se divide el conjunto de señales de entrada en dos grupos: el conjunto de entrenamiento y el conjunto de pruebas. El conjunto de entrenamiento estará formado por el 80% de las señales, seleccionadas de modo aleatorio, y el conjunto de pruebas por el 20% restante.

Para los ejemplos mostrados se calcula la acuracidad y la matriz de confusión.

* La **acuracidad** es el porcentaje de ejemplos clasificados correctamente.
* La **matriz de confusión** compara la clase en la que el modelo clasifica la señal, con la clase real a la que pertenece. Para clasificación binaria se tendrá una matriz de dimensión $2 \\times 2$.
"""
st.markdown(texto1)