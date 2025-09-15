# Modelo de lenguaje generativo para tutor en sistemas de comunicaciones digitales

### Trabajo Fin de Máster. Máster en Ingeniería de Telecomunicación

Sergio García López. Universidad de Sevilla. 2025

## Resumen

En los últimos años, los denominados LLMs (Large Language Models) han transformado los campos de la 
inteligencia artificial y del procesamiento del lenguaje natural. Estos sistemas, basados en arquitecturas 
neuronales avanzadas como el modelo Transformer y entrenados con cantidades masivas de datos, han 
demostrado una capacidad sin precedentes para generar textos coherentes y contextualizados. Dichos modelos 
han redefinido aplicaciones prácticas en áreas como la traducción automática, la atención al cliente, la 
clasificación de documentos e incluso la asistencia en investigación científica. Sin embargo, su rápido desarrollo 
plantea desafíos críticos en términos de ética, sesgos de entrenamiento, sostenibilidad computacional y 
adaptación a contextos específicos. 

Dentro de este marco, este Trabajo Fin de Máster propone adaptar un LLM de código abierto preentrenado, 
como es el caso de Llama 3.1, mediante técnicas RAG (Retrieval Augmented Generation) para que actúe como 
un tutor especializado en el ámbito de los sistemas de comunicaciones digitales. Para ello, se utilizará una 
arquitectura basada en LangChain y ChromaDB, con el objetivo de realizar búsquedas semánticas sobre una 
base de datos vectorial construida a partir de documentos académicos del área de interés. Dichos documentos 
serán segmentados en fragmentos de texto y vectorizados mediante un modelo de embedding del framework 
Hugging Face para almacenarlos en ChromaDB. Tras realizar la búsqueda semántica, los fragmentos más 
relevantes se incorporarán dinámicamente al prompt del LLM, con el fin de generar respuestas contextualizadas 
y alineadas con la consulta. Además, se empleará la librería Transformers de Hugging Face tanto para la 
ejecución local del modelo como para aprovechar sus herramientas de cuantización, permitiendo reducir 
significativamente el tiempo de inferencia sin comprometer en exceso la precisión de las respuestas generadas. 

<img width="1920" height="1080" alt="Arquitectura_TFM" src="https://github.com/user-attachments/assets/18735142-ff37-44fa-ad94-19780108c57c" />

## Backend

El backend constituye el núcleo del sistema y se encarga de integrar todos los procesos necesarios para ofrecer la funcionalidad de un asistente experto en sistemas de comunicaciones digitales. A través de una arquitectura modular, el backend centraliza la gestión de la base de conocimiento, la orquestación del flujo RAG y la inferencia del modelo Llama 3.1, exponiendo estos servicios de manera accesible mediante una API REST. Esta organización permite mantener separadas las responsabilidades de cada componente, asegurando escalabilidad, mantenibilidad y la posibilidad de ampliar el sistema con nuevas capacidades.

Los principales elementos del backend son:

- **API REST:** Implementada con la librería FastAPI, esta API actúa como punto de entrada para las solicitudes del cliente, ya sean consultas al modelo de lenguaje o modificaciones en la base de datos vectorial. Este componente abstrae los servicios ofrecidos en un conjunto de URLs y métodos HTTP (endpoints).

- **Modelo de embedding:** Este elemento se encarga de convertir texto a una representación vectorial para que pueda ser almacenado en una base de datos vectorial. Para ello, se utiliza el modelo `sentence-transformers/all-mpnet-base-v2`, un modelo de embedding de código abierto que puede descargarse desde Hugging Face para utilizarse de forma local. Previamente, los documentos que se desean vectorizar deben someterse a un proceso de limpieza y segmentación, con el objetivo de maximizar el rendimiento del sistema RAG tanto en la búsqueda y recuperación de información como en la generación de respuestas del modelo de lenguaje.

- **ChromaDB:** Esta base de datos vectorial es la encargada de almacenar las representaciones vectoriales de los documentos. Además, incluye un mecanismo de búsqueda por similitud que permite identificar y recuperar los fragmentos de texto más relevantes para la consulta de un usuario. En este proyecto, la métrica que se utiliza para realizar búsquedas es la similitud coseno.

- **LangChain:** Este framework se utiliza para coordinar los procesos y los flujos de información intercambiados entre los elementos del sistema RAG, desde la recuperación de contexto de la base de datos vectorial hasta su incorporación en la estructura del prompt siguiendo las especificaciones de Meta para el modelo Llama 3.1 Instruct. LangChain actúa como un orquestador, implementando cadenas (chains) que encapsulan la lógica de recuperación y generación.

- **Modelo Llama 3.1:** Desplegado de forma local mediante la integración con Hugging Face, este modelo procesa el prompt enriquecido con el contexto recuperado y genera la respuesta para el usuario. Además, está optimizado con técnicas de cuantización en 8 bits para reducir el consumo de memoria y el tiempo de inferencia, permitiendo su ejecución en entornos con recursos de hardware limitados sin comprometer significativamente la calidad de la respuesta.

### Flujo de funcionamiento del sistema RAG

El backend implementa un flujo de trabajo basado en **Retrieval-Augmented Generation (RAG)**, que permite enriquecer las respuestas del modelo con información contextual recuperada de la base de conocimiento.

Las etapas principales son:

1.	**Preprocesamiento, vectorización y almacenamiento de documentos**: Los documentos que conforman la base de conocimiento, tras someterse a un proceso de limpieza y segmentación, son procesados mediante el modelo de embedding, generando representaciones vectoriales que son almacenadas en ChromaDB. En esta etapa debemos asegurarnos de que cada vector mantenga la semántica necesaria para ser identificado correctamente durante la fase de búsqueda.
   
2.	**Consultas y recuperación de información de contexto**: Cuando se recibe una pregunta del usuario vía API, esta se vectoriza con el modelo de embedding, y mediante búsqueda por similitud se consulta la base de datos para recuperar los fragmentos más relevantes. El número de fragmentos recuperados y su relevancia influyen directamente en la calidad de la respuesta final, por lo que estos parámetros deben ajustarse cuidadosamente.
   
3.	**Construcción del prompt**: Tras realizar la búsqueda en la base de datos, los fragmentos recuperados se integran junto con la consulta original en un prompt estructurado. Esta integración sigue el formato oficial de prompts de Meta para Llama 3.1, garantizando que el modelo interprete correctamente las instrucciones del system prompt, las consultas del usuario y los mensajes previos que se hayan podido intercambiar en la conversación.
   
4.	**Generación de la respuesta del modelo de lenguaje**: El prompt formado por la consulta del usuario junto a información de contexto se envía al modelo Llama 3.1 para que genere la respuesta final.

<img width="1920" height="1080" alt="RAG_TFM" src="https://github.com/user-attachments/assets/daba14b6-82c8-45e8-8ab5-55d2a679c520" />

### Endpoints de la API

Los endpoints de la API se organizan en grupos distintos en función de las funcionalidades que ofrecen:

- **Models**
  - **GET /models:** Devuelve la lista de modelos disponibles en el sistema. Actualmente, solo se utiliza el modelo Llama 3.1 8B Instruct, pero se podría ampliar el sistema con más modelos Llama o de otras familias, como Mistral o Qwen.  
  - **GET /models/{model_id}:** Permite obtener información de un modelo concreto identificado por el parámetro `model_id`. En el caso del modelo Llama utilizado en este proyecto, su identificador es `llama-3.1-8b-instruct`.

- **Chat**
  - **POST /chat/completions:** Permite enviar mensajes al modelo de lenguaje y recibir la respuesta generada. La solicitud se realiza mediante un JSON que contiene una lista de mensajes, donde cada mensaje especifica un rol (`system`, `user` o `assistant`) y su contenido textual. Además, la solicitud incluye una opción para habilitar el modo RAG del sistema, mejorando así la precisión y relevancia de la respuesta generada.

- **Documents**
  - **GET /docs:** Devuelve la lista de documentos PDF almacenados en el sistema RAG.  
  - **PUT /docs:** Permite subir un nuevo documento PDF al sistema RAG, que será procesado, dividido en fragmentos y vectorizado para futuras consultas.  
  - **GET /docs/{filename}:** Permite descargar un documento PDF almacenado en el sistema, identificado por el parámetro `filename`.  
  - **DELETE /docs/{filename}:** Elimina un documento PDF y sus vectores asociados de la base de datos vectorial. Para localizar el documento se utiliza el parámetro `filename`.

<img width="586" height="506" alt="api_endpoints" src="https://github.com/user-attachments/assets/13c7cb55-2048-4618-9632-aebd56517926" />

### Validación del asistente

El propósito del dataset de validación es proporcionar un conjunto representativo de preguntas y respuestas que permita medir de forma objetiva la capacidad del asistente. El dataset está compuesto por 169 preguntas extraídas y generadas a partir de documentación técnica y académica, y en particular, de los apuntes de clase de la asignatura “Sistemas de Comunicaciones” del Máster en Ingeniería de Telecomunicación de la Universidad de Sevilla, impartida por el catedrático Juan José Murillo Fuentes. Estas preguntas se han recopilado y estructurado por bloques temáticos, procurando que su distribución refleje de manera proporcional la relevancia, extensión y profundidad que cada tema presenta en los documentos de referencia.

<img width="900" height="600" alt="questions_per_topic" src="https://github.com/user-attachments/assets/68b1a7af-9ea4-4b96-8e95-704a1d1113b6" />

Los resultados obtenidos tras la evaluación del asistente permiten llevar a cabo un análisis detallado de su desempeño en preguntas relacionadas con las comunicaciones digitales. En términos globales, el modelo alcanza una precisión del 70,41%, lo que evidencia que, en promedio, es capaz de responder correctamente a más de dos tercios de las preguntas presentes en el dataset de validación. Este resultado refleja un rendimiento notable para un modelo Llama 3.1 8B Instruct apoyado en un sistema RAG básico, lo que confirma su potencial como herramienta de apoyo para la enseñanza y la resolución de problemas sencillos en el ámbito de las comunicaciones digitales. No obstante, los resultados también muestran las limitaciones inherentes a la naturaleza probabilística de los modelos de lenguaje y al reducido tamaño del modelo empleado. En comparación con arquitecturas de mayor escala, como Llama 3.1 405 B o los modelos GPT de OpenAI, un modelo de 8 mil millones de parámetros dispone de menor capacidad para abordar tareas que requieren un razonamiento complejo y muestra una mayor propensión a generar errores y alucinaciones.

<img width="900" height="600" alt="accuracy_per_topic" src="https://github.com/user-attachments/assets/32e25ae2-9ecf-452c-b5fe-6a29444fd20d" />

### Despliegue del backend con Docker Compose

Para desplegar el backend como contenedor Docker, debemos situarnos en el directorio donde se encuentra el fichero `docker-compose.yml` y ejecutar:

```bash
docker compose up
```

Para ejecutarlo en segundo plano, podemos utilizar el comando:

```bash
docker compose up -d
```

Finalmente, para detener y eliminar el contenedor podemos usar:

```bash
docker compose down
```
