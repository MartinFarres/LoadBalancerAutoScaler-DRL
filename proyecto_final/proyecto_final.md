# Introducción

En el despliegue de microservicios y aplicaciones modernas, la eficiencia operativa depende de la capacidad del sistema para adaptarse a la demanda variable. Docker se ha consolidado como herramienta fundamental para la contenerización y gestión de entornos de desarrollo y producción debido a su simplicidad y portabilidad. Sin embargo, un reto dentro de entornos basados en estas herramientas es el autoescalado, que es la capacidad de ajustar dinámicamente el número de contenedores en ejecución para responder a picos de tráfico o carga de procesamiento. 

Se ha empleado aprendizaje por refuerzo para el entrenamiento del agente ya que le permitirá aprender a optimizar sus acciones en un entorno cambiante maximizando las recompensas obtenidas, permitiéndole aumentar o reducir la cantidad de contenedores disponibles, balanceando la carga entre ellos y evitando así la saturación de servicios.

Este tipo de problemas cuentan con la dificultad de tener que obtener en tiempo real métricas correspondientes a los contenedores como el uso de CPU, memoria RAM, latencia de red y el nivel de actividad presente en los contenedores. Para simular el entorno de ejecución se ha utilizado la Biblioteca Gymnasium de OpenAI con un entorno personalizado para emplear las métricas necesarias, el entorno observado y una función de recompensa.


## Marco teórico

Se deberá poner especial énfasis en aquellos elementos que van a utilizarse para proponer una implementación. Incluir una descripción teórica y general del funcionamiento del (o los) algoritmos y sus principales elementos propuestos para lidiar con el problema elegido.  
Como así también justificar debidamente la elección de dicho algoritmo. Consultar bibliografía externa, la cual deberá estar debidamente citada.

El autoescalado en arquitecturas de microservicios se basa en la capacidad de replicar unidades de ejecución de forma aislada. Docker permite esta encapsulación mediante contenedores que comparten el kernel del sistema operativo host, lo que garantiza un levantamiento de instancias mucho más veloz que las máquinas virtuales tradicionales.

### Aprendizaje por Refuerzo (Reinforcement Learning)

El Aprendizaje por Refuerzo (RL) es un paradigma del aprendizaje automático centrado en el entrenamiento de un agente capaz de tomar decisiones secuenciales en un entorno dinámico con el objetivo de maximizar una recompensa acumulada a largo plazo. A diferencia del aprendizaje supervisado, el agente no recibe ejemplos de "decisiones correctas", sino que aprende a través de un proceso iterativo de prueba y error, evaluando el impacto de sus acciones en el estado del sistema.

Este proceso de aprendizaje se realiza mediante los siguientes elementos fundamentales:
- **Agente:** Es la entidad lógica que observa el sistema y ejecuta las acciones de escalado.
- **Entorno (Environment):** Representa el cluster de contenedores y la infraestructura de Docker donde operan los servicios.
- **Estado ($S$):** Es la representación cuantitativa de la situación actual, como los porcentajes de uso de CPU y memoria reportados por la API de Docker.
- **Acciones ($A$):** El conjunto de decisiones disponibles para el agente, específicamente: incrementar réplicas, reducir réplicas o mantener el estado actual.
- **Recompensa ($R$):** Una señal escalar que retroalimenta al agente. Una recompensa positiva indica una gestión eficiente de recursos, mientras que una negativa puede señalar saturación del servicio o desperdicio de hardware.
- **Política ($\pi$):** La estrategia o "mapeo" que el agente sigue para determinar qué acción tomar ante un estado determinado.El objetivo final del entrenamiento es hallar una política óptima ($\pi^*$) que garantice la estabilidad del cluster bajo cualquier escenario de carga. 

Para alcanzar este nivel de optimización en entornos de alta dimensionalidad, se recurre a algoritmos avanzados como Q-Learning, Deep Q-Network (DQN) y Proximal Policy Optimization (PPO).

### Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) es un algoritmo de aprendizaje por refuerzo encuadrado en los métodos de policy gradient. 

A diferencia de los algoritmos basados en valores (como Q-Learning), PPO optimiza directamente la política del agente, mediante una red neuronal que produce una distribución de probabilidades sobre el espacio de acciones para cada estado observado.Este algoritmo pertenece a la familia de los métodos Actor-Critic, donde el aprendizaje se divide en dos estructuras complementarias:
- **Actor:** Una red neuronal encargada de generar la política $\pi(a|s)$, determinando la probabilidad de seleccionar cada acción ante un estado dado.
- **Critic:** Una red que estima la función de valor $V(s)$, proporcionando una evaluación del estado actual que sirve para calcular la ventaja (advantage), guiando así la dirección y magnitud de las actualizaciones del actor.

La principal innovación de PPO radica en su función objetivo de proximidad (clipped surrogate objective). Está diseñada para resolver el problema de la inestabilidad en el entrenamiento, evitando que la política sufra cambios bruscos entre iteraciones. 
Para ello, el algoritmo calcula un ratio de probabilidad entre la política nueva y la anterior:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

En lugar de maximizar este ratio sin restricciones, PPO aplica un recorte que limita el valor de $r_t(\theta)$ dentro de un rango determinado. Este proceso asegura que las actualizaciones sean pequeñas y controladas, manteniendo la nueva política "cerca" de la anterior, garantizando una convergencia más estable y robusta.


### Justificacion

Para este proyecto se ha decidido por utilizar el algoritmo PPO

# Diseño Experimental

Se deberá presentar una sección en donde se describa todo el proceso realizado para poner a prueba el o los algoritmos utilizados. Esto deberá incluir primeramente:

- Las métricas consideradas a fin de establecer el alcance y rendimiento del algoritmo sobre el problema dado.
- Las herramientas utilizadas para la implementación como así también para medir el rendimiento del algoritmo (frameworks, simuladores, etc.).
- En aquellos casos en donde resulte adecuado, se deberá indicar todo el proceso realizado para la obtención y adecuación del conjunto de datos.
- Un detalle y justificación de los experimentos realizados a fin de determinar los resultados. Este deberá incluir tablas y/o gráficos que resuman los resultados.

Evitar incluir en esta sección código fuente. Este se puede incluir como un apéndice al final del documento. Si es posible, incluir pequeños fragmentos de pseudo-código de ser necesario.

# Análisis y discusión de resultados

En esta sección se deberá realizar un mínimo análisis sobre los resultados obtenidos.  
El objetivo es tratar de razonar sobre las causas de los resultados obtenidos en la fase experimental a fin de proveer una posible justificación.  
Aquí se incluyen posibles limitaciones en los algoritmos elegidos, en la simulación planteada, los datos, etc.

# Conclusiones finales

Observaciones finales sobre el tema y es muy importante indicar aquellas tareas o experimentos que quedaron sin realizar, pero que eventualmente podrían realizarse en el futuro.

# Bibliografía

Incluir la bibliografía utilizada para el trabajo. Es importante referenciar en el cuerpo del trabajo las diferentes fuentes utilizadas.

**Ejemplo:**

[1] Barrat, A., Barthelemy, M., & Vespignani, A. (2008). _Dynamical processes on complex networks_. Cambridge University Press.  
[2] Bengio, Y., Courville, A., & Vincent, P. (2013). _Representation learning: A review and new perspectives_. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8), 1798–1828.
