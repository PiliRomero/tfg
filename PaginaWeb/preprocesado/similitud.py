import streamlit as st

#################################
# Medidas de similitud
#################################

st.title("Medidas de similitud")

texto1=r"""
Para la aplicación de algunas técnicas de aprendizaje automático, como por ejemplo el aprendizaje por agrupación, es necesario definir medidas de similitud entre vectores de datos.
Se puede definir la similitud entre los vectores $u$ y $v$ como:
$$
s(u,v) = \frac{1}{1+ \sqrt{\sum_{i=1}^{n}(u_{i}-v_{i})^{2}}}
$$
De la fórmula anterior se deduce que, si la distancia euclídea entre los vectores $u$ y $v$ es cero, entonces la similitud entre $u$ y $v$, $s(u,v)$ es igual a 1. Por otra parte, según aumenta la distancia euclídea entre $u$ y $v$ disminuye la similitud $s(u,v)$.

Esta medida se puede ver afectada por el rango de valores de los vectores, por eso puede ser necesario normalizarlos previamente. Por ejemplo, para **normalizar** $v$ se resta a cada elemento de $v$ su media ($\mu_{v}$) y se divide entre la desviación típica ($\sigma_{v}$) :
$$
\frac{v_{i}-\mu_{v}}{\sigma_{v}}
$$

Otra de las medidas de similitud utilizadas es la **similitud del coseno**, que se calcula como el valor absoluto del coseno del ángulo que forman los vectores $u$ y $v$ :

$$
Similitud_{coseno}(u,v) = \frac{\left| u \cdot v \right|}{\| u\| \|v \|} = \frac{\left| \sum_{i=1}^{n}u_{i}v_{i}\right|}{\left ( \sqrt{\sum_{i=1}^{n}u_{i}^{2}} \right )\left ( \sqrt{\sum_{i=1}^{n}v_{i}^{2}} \right )}
$$

"""
st.markdown(texto1)