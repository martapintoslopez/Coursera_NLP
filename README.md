## Coursera - NLP 
# Proyecto de Análisis de Texto para Clasificación de Cursos en Ciencias y Humanidades

Este proyecto tiene como objetivo clasificar cursos en dos categorías: Ciencias y Humanidades. Se utiliza un enfoque de procesamiento de lenguaje natural (NLP) y aprendizaje automático para analizar el título de los cursos y predecir su categoría.

## Descripción del Código

El código proporcionado en este repositorio consta de varios archivos Python que realizan diferentes tareas en el proceso de clasificación de cursos:

- `EDA.ipynb`: Notebook que realiza una limpieza de los datos.
- `functions.py`: Contiene funciones para preprocesar los títulos de los cursos, incluyendo la eliminación de enlaces, tokenización, eliminación de palabras vacías y puntuación, y reducción de palabras a su raíz, además de funciones auxiliares para construir tablas de frecuencia, entrenar modelos de regresión logística y Naive Bayes, y realizar predicciones.
- `training_predictions.ipynb`: Notebook que entrena el modelo con regresión logistica para hacer predicciones de clasificación.
- `Classifying_Naive_Bayes.ipynb`: Notebook que entrena el modelo Naive Bayes para hacer predicciones de clasificación.
- `autocorrect.ipynb`: Notebook donde estoy desarrollando un autocorrector.

## Configuración del Entorno

Para ejecutar este código, asegúrate de tener instaladas todas las dependencias necesarias recogidas en el archivo requirements.txt. Puedes instalarlas ejecutando el siguiente comando: pip install -r requirements.txt

## Datos

Los datos utilizados en este proyecto se encuentran en el archivo `DatasetFinal.csv`. Este archivo contiene información sobre diferentes cursos.

### Descripciones de las Columnas:

* Course Title (Título del Curso): Esta columna contiene el título del curso ofrecido en Coursera.
* Rating (Calificación): La columna de calificación probablemente contiene la calificación promedio del curso, según lo proporcionado por los usuarios que lo han completado. Las calificaciones suelen darse en una escala, como de 1 a 5 estrellas.
* Level (Nivel): Esta columna indica el nivel de dificultad o complejidad del curso. Podría categorizar los cursos como principiante, intermedio o avanzado, por ejemplo.
* Schedule (Horario): Esta columna puede especificar el horario o la programación del curso, como si tiene un horario flexible o es de aprendizaje práctico.
* What you will learn (Lo que aprenderás): Esta columna probablemente describe los objetivos de aprendizaje o los temas cubiertos en el curso. Proporciona un resumen del conocimiento o habilidades que los participantes pueden esperar adquirir.
* Skill gain (Adquisición de habilidades): Esta columna puede detallar las habilidades específicas que los participantes adquirirán al completar el curso.
* Modules (Módulos): La columna de módulos probablemente enumera las diferentes secciones o unidades que componen el curso. Podría proporcionar una visión general de la estructura y organización del curso.
* Instructor (Instructor): Esta columna contiene información sobre el(s) instructor(es) o conferenciantes que imparten el curso.
* Offered By (Ofrecido por): Esta columna probablemente especifica la institución u organización que ofrece el curso en la plataforma Coursera.
* Keyword (Palabra clave): Esta columna puede contener palabras clave o etiquetas asociadas con el curso, que pueden ayudar a los usuarios a buscar cursos relevantes basados en temas o temas específicos.
* Course Url (URL del Curso): Esta columna probablemente contiene la URL o enlace web a la página del curso en la plataforma Coursera.
* Duration to complete (Approx.) (Duración para completar (Aprox.)): Esta columna especifica el tiempo aproximado necesario para completar el curso. Se da en horas.
* Number of Reviews (Número de Reseñas): Esta columna contiene el recuento de reseñas o calificaciones enviadas por usuarios que han completado el curso. Proporciona una indicación del nivel de popularidad y satisfacción del usuario del curso.

*Nota: Inspirado en NLP Specialization de DeapLearning.AI.*
