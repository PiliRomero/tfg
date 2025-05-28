import streamlit as st
st.title("SVM para clasificación binaria")

st.header("Datos separables linealmente")


texto1=r"""

Las máquinas de vectores soporte (SVM) son algoritmos de aprendizaje automático supervisado utilizados principalmente para resolver problemas de clasificación (binaria y multiclasificación) y regresión. Aquí se emplearán únicamente para clasificación.

Se parte de un conjunto de datos de entrenamiento etiquetados: $ \left\{ (x_{1}, y_{1}) , \cdots , (x_{n},y_{n})\right\}$, 
donde cada ejemplo de entrenamiento $x_{i}\; \epsilon\; \mathbb{R}^{p}$ e $y_{i}$ es la correspondiente etiqueta que puede tomar 
los valores $\left\{ -1,+1\right\}$, en función de la clase a la que pertenezca la observación $x_{i}$.

Si el conjunto de datos de entrenamiento son linealmente separables en dos clases $\left\{ -1,+1\right\}$, entonces 
existe un hiperblano 

$$
\pi = \left\{ x\; \epsilon\;  \mathbb{R}^{p}  \; | \;\omega^{t}x + b =0 \right\}
$$

 con $\omega=(\omega_{1}, \dots \omega_{p})^{t} \; \epsilon \;\mathbb{R}^{p}$ y b es una contante sin determinar, 
 de modo que divide el conjunto de datos de entrenamiento en dos clases:  


* $\pi_{+} = \left\{ x\; \epsilon\;  \mathbb{R}^{p}  \; | \;\omega^{t}x + b \geq 0 \right\}$
* $\pi_{-} = \left\{ x\; \epsilon\;  \mathbb{R}^{p}  \; | \;\omega^{t}x + b < 0 \right\}$


$$
y= \left\{ \begin{array}{rlc} 1 & si & x \; \epsilon \; \pi_{+} \\ -1 & si & x \; \epsilon \; \pi_{+}  \end{array} \right.
$$

La idea que subyace en el las máquinas de vectores soporte es tratar de definir un hiperplano de separación único y óptimo. 
Para ello se define el margen de un hiperplano de separación $\tau$ como la distancia del hiperplano al ejemplo más cercano. 
El hiperplano será óptimo si su margen es de tamaño máximo. Entonces, teniendo en cuenta que la distancia de un punto $x_{i}$ 
al hiperplano es igual a $|\omega^{t}x_{i} + b|$ y que $y_{i}(\omega^{t}x_{i} + b)>0$, entonces:

$$
Distancia(\pi,x_{i}) \geq \tau \; \Leftrightarrow \frac{\left | \omega^{t}x_{i} + b \right | }{\left\| \omega \right\|} \geq \tau \Leftrightarrow  \frac{ y_{i}(\omega^{t}x_{i} + b)}{\left\| \omega \right\|} \geq \tau, \: i=1, \dots, n
$$

donde $\omega$ es el vector normal del hiperplano $\pi$.

Para los puntos más cercanos al hiperplano, que delimitan el margen de cada lado y que se denominan 
**vectores soporte**, se verifica que:

$$
\frac{ y_{i}(\omega^{t}x_{i} + b)}{\left\| \omega \right\|} = \tau
$$
"""
st.write(texto1)
st.image("./PaginaWeb/imagenes/hiperplano1.jpg")

texto2=r"""Entonces, teniendo en cuenta la ecuación de la distancia de $\pi$ a $x_{i}$, maximizar el margen del hiperplano de separación 
equivale a minimizar $\left\| \omega \right\|$ (o equivalentemente $\frac{1}{2}\left\| \omega \right\|^2 = \frac{1}{2}\omega^{t}\omega$). 
Para limitar el número de soluciones posibles a una única solución, se añade la restricción $\tau \cdot \left\| \omega \right\| = 1$.

Por lo tanto, el problema de optimización se reduce a minimizar $\frac{1}{2}\left\| \omega \right\|^2$ sujeto a $\frac{y_{i}(\omega^{t}x_{i} + b)}{\left\| \omega \right\|} \geq \tau, \: i=1, \dots, n$ o equivalentemente:

$$
 \underset{\omega}{min} \; \frac{1}{2} \left\| \omega \right\|^{2}
$$
$$
s.a.: \; y_{i}(\omega^{t}x_{i} + b)-1 \geq 0, \; i = 1, \dots, n
$$


Para resolver el problema de minimización con restricciones se construye la función Lagrangiana:

$$
\mathfrak{L}(\omega,b,\alpha) =  \frac{1}{2}\omega^{t}\omega - \sum_{i=1}^{n}\alpha_{i}(y_{i}(\omega^{t}x_{i} + b)-1) 
$$


donde $\alpha_{i}$ son los multiplicadores de Lagrange. Se aplica el teorema de Karush-Kuhn-Tucker y aplicando las derivadas parciales respecto a los parámetros de interés en la se llega a:

$$
\frac{ \partial \mathfrak{L}(\omega,b,\alpha)}{\partial \omega} = \omega - \sum_{i=1}^{n}\alpha_{i}y_{i}x_{i} = 0 \Leftrightarrow \omega = \sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}
$$
$$
\frac{ \partial \mathfrak{L}(\omega,b,\alpha)}{\partial b}= \sum_{i=1}^{n}\alpha_{i}y_{i} = 0 
$$

$$\alpha_{i}(y_{i}(b+\omega^{t}x_{i})-1) = 0 \; \forall \;i=1, \dots , n$$

Substituyendo estos resultados en la función Lagrangiana se construye el problema dual:
$$
 \underset{\alpha}{max} \; \sum_{i=1}^{n} \alpha_{i} - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_{i} \alpha_{j} y_{i}y_{j}x_{i}^{t},x_{j}\\
 s.a.: \; \sum_{i=1}^{n} \alpha_{i} y_{i} = 0 \\
 \alpha_{i} \geq 0, \; i=1, \dots, n
$$


Una vez se obtiene la solución óptima del problema dual $\hat{\alpha} = (\hat{\alpha_{1}}, \dots, \hat{\alpha_{n}})$ se obtiene la solución el problema primal:

$$
\hat{\omega} = \sum_{i=1}^{n}\hat{\alpha_{i}} y_{i}x_{i}
$$


entonces reemplazando $\omega$ por $\hat{\omega}$  se obtiene el hiperplano de separación como:

$$
\pi = \left\{ x\; \epsilon\;  \mathbb{R}^{p}  \; | \;(\hat{\omega})^{t}x + \hat{b} =0 \right\} = \left\{ x\; \epsilon\;  \mathbb{R}^{p}  \; |  \sum_{i=1}^{n}\hat{\alpha_{i}} y_{i}x^{t}x_{i} + \hat{b} = 0\ \right\}; 
$$

Para determinar el parámetro $\hat{b}$ se elije un vector soporte $(x_{vs},y_{vs})$ y se calcula

$$
\hat{b} = y_{vs}-(\hat{\omega})^{t}x_{vs}
$$

o bien promediando todos los vectores soporte. 

En resumen, cuando los datos son separables linealmente se elige como hiperplano de separación el que equidista de los dos datos más cercanos de cada una de las dos clases. Además, los parámetros del hiperplano se determinan teniendo en cuenta únicamente los vectores soporte (aquellos $(x_{i},y_{i})$ en los que $y_{i}(\omega^{t}x_{i} + b)=1$).

En la práctica el conjunto de datos de entrenamiento raramente va a ser linealmente separables debido a la existencia de ruido.

"""
st.write(texto2)