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

Para este proyecto se ha decidido utilizar el algoritmo PPO:
-  **DQN** originalmente trabaja con estados discretos, a diferencia de **PPO** tiene una mejor reaccion ante metricas continuas como uso porcentual 
-  La politica de recorte para que los cambios no sean tan abruptos permiten que las acciones del cluster no oscilen levantando y dando de baja muchos contenedores continuamente 
-  Mientras que **DQN** se limita a estimar el valor de una accion, **PPO**, utiliza una arquitectura Actor-Critic, lo cual permite que el agente aprenda no solo a maximizar el rendimiento, sino a reducir la varianza, lo que se traduce en un escalado mucho más predecible y menos errático.



# Diseño Experimental

Se deberá presentar una sección en donde se describa todo el proceso realizado para poner a prueba el o los algoritmos utilizados. Esto deberá incluir primeramente:

- Las métricas consideradas a fin de establecer el alcance y rendimiento del algoritmo sobre el problema dado.
- Las herramientas utilizadas para la implementación como así también para medir el rendimiento del algoritmo (frameworks, simuladores, etc.).
- En aquellos casos en donde resulte adecuado, se deberá indicar todo el proceso realizado para la obtención y adecuación del conjunto de datos.
- Un detalle y justificación de los experimentos realizados a fin de determinar los resultados. Este deberá incluir tablas y/o gráficos que resuman los resultados.

Evitar incluir en esta sección código fuente. Este se puede incluir como un apéndice al final del documento. Si es posible, incluir pequeños fragmentos de pseudo-código de ser necesario.

## Métricas de Desempeño
Para evaluar la efectividad del autoscaler y el comportamiento del algoritmo PPO, se han definido cinco métricas principales que permiten medir tanto la calidad del servicio como la eficiencia de los recursos:

- **Uso de la CPU (cpu_usg):** Uso de CPU por contenedor, normalizado para identificar la carga computacional.

- **Uso de memoria RAM (ram_usg_pct y ram_total_normalize):** Representan el consumo porcentual y absoluto de memoria RAM, permitiendo al agente distinguir entre picos momentáneos y riesgo de saturación (OOM).

- **Latencia (latency):** Tiempo de respuesta de las peticiones, métrica crítica para la experiencia de usuario.

- **Ratio de Errores (error_rate):** Porcentaje de respuestas fallidas (ej. HTTP 5xx), que indica si el cluster está sobrepasado.

- **Estado (status):** Indicador binario o categórico de la disponibilidad del servicio.

## Herramientas Utilizadas

Para realizar las pruebas y poner en marcha el sistema, se usaron las siguientes herramientas:

- **Docker:** Se usó para crear los contenedores de los microservicios. Esto asegura que la aplicación corra siempre igual y que podamos medir exactamente cuánta CPU y RAM gasta cada parte.

- **Docker-Compose:** Se utilizó para manejar el conjunto de los contenedores. Es la pieza clave que permite al algoritmo subir o bajar la cantidad de instancias (escalar) de forma sencilla mediante comandos.

- **Stable Baselines3:** Es el framework de aprendizaje por refuerzo basado en PyTorch que proporciona la infraestructura necesaria para la ejecución de PPO. Fue seleccionado por su robustez y por ofrecer implementaciones optimizadas que garantizan la estabilidad de los gradientes durante el entrenamiento del agente.

- **OpenAI Gymnasium:** Es la interfaz estándar utilizada para abstraer el cluster de Docker como un entorno de aprendizaje. Proporciona el marco necesario para simular estados y ejecutar acciones, permitiendo así el entrenamiento sistemático del agente.

- **Pydantic:** Se usó para definir y validar los modelos de datos de las métricas (ContainerMetrics) y las acciones (AgentAction). Esto sirve para que, si una medición de Docker viene mal o incompleta, el sistema no falle y los datos siempre tengan el formato correcto antes de entrar al algoritmo.

- **Locust:** Se utilizó como la herramienta de generación de carga para simular el tráfico de usuarios reales sobre el sistema. Su función fue poner a prueba al modelo en un entorno de "cluster funcional", permitiendo observar cómo responde el agente ante demandas de tráfico auténticas en lugar de depender únicamente de datos simulados 

## Proceso de entrenamiento

Para el entrenamiento del agente no nos basamos en datos estaticos preestablecidos si no que se opto por el uso de datos generados de manera aleatoria para mejorar los resultados del proceso de aprendizaje.

### Generación de datos y entornos de entrenamiento
Para el desarrollo del agente, se definieron dos fuentes de datos distintas que permitieron una evolución progresiva del aprendizaje:

- *Entorno Simulado:* En la fase inicial de entrenamiento, se utilizaron funciones matemáticas con ruido gaussiano para generar señales de carga sintéticas. Este enfoque permitió simular comportamientos estocásticos del sistema, proporcionando al agente un entorno controlado pero variable donde aprender las políticas de escalado sin depender de la infraestructura física acelerando el proceso de preentrenamiento.

- *Entorno Real con Locust:* Una vez que el agente demostró estabilidad en la simulación, se pasó a un "cluster funcional". En esta etapa, se utilizó Locust para generar tráfico de usuarios auténtico. Esto permitió recolectar métricas de rendimiento reales extraídas de la API de Docker, enfrentando al agente a la latencia real de red y a los tiempos de respuesta del motor de contenedores.



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
