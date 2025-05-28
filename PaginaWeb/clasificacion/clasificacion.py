import streamlit as st
st.title("Máquinas de Vectores Soporte (SVM)")

texto1="""
Las máquinas de vectores soporte (SVM) son algoritmos de aprendizaje automático supervisado utilizados principalmente para resolver problemas de clasificación (binaria y multiclasificación) y regresión.

En esta página se utilizarán las SVM en primer lugar para la clasificación binaria de señales digitales y, posteriormente, se ampliará el análisis para la clasificación en más de dos grupos.

Para evaluar el rendimiento del modelo de la clasificación se divide el conjunto de señales de entrada en dos grupos: el conjunto de entrenamiento y el conjunto de pruebas. El conjunto de entrenamiento estará formado por el 80% de las señales, seleccionadas de modo aleatorio, y el conjunto de pruebas por el 20% restante.

Para los ejemplos mostrados se calcula la acuracidad y la matriz de confusión.

* La **acuracidad** es el porcentaje de ejemplos clasificados correctamente.
* La **matriz de confusión**  resume los datos reservados para el test según su etiqueta real y la  clase a la que fueron asignadas según el modelo.. Para clasificación binaria se tendrá una matriz de dimensión $2 \\times 2$.

La acuracidad puede no ser adecuada cuando se trabaja con datos desequilibrados. Por ejemplo, en clasificación binaria, si en el conjunto de señales de una de las clases es mucho más numerosa que la otra, puede ocurrir que el modelo utilizado prediga mucho mejor la clase mayoritaria. El modelo podría llegar a predecir todos los datos del test como pertenecientes a la clase mayoritaria y la acuracidad seguir mostrando un valor muy elevado. 
"""
st.markdown(texto1)