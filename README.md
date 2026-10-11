# Movie Recommendation System

API experimental de exploración y recomendación de películas construida con **Python, FastAPI y scikit-learn**. El repositorio contiene trabajos de extracción/transformación de datos y un recomendador basado en similitud de contenido.

> **Estado:** proyecto académico/exploratorio. Se han inspeccionado los archivos fuente; no se ha comprobado el despliegue ni la ejecución desde un entorno limpio.

## Qué hace el proyecto

- Consultas por idioma, duración, franquicia, país, productora y director.
- Construcción de características textuales a partir de descripciones, géneros y otros atributos.
- Vectorización **TF-IDF** con unigramas y bigramas.
- Recomendación de títulos similares utilizando **NearestNeighbors** con métrica de distancia coseno.

## Flujo técnico

~~~mermaid
flowchart LR
    A["Datos de películas"] --> B["ETL y preparación"]
    B --> C["Características de contenido"]
    C --> D["TF-IDF"]
    D --> E["Vecinos más cercanos<br/>Distancia coseno"]
    E --> F["API FastAPI<br/>Recomendaciones"]
~~~

## Archivos principales

| Ruta | Propósito |
| --- | --- |
| [main.py](main.py) | API y método de recomendación. |
| [ETL](ETL) | Material de transformación de datos. |
| [EDA](EDA) | Material de análisis exploratorio. |
| [ML](ML) | Material de aprendizaje automático. |
| [datasetLimpio](datasetLimpio) | Directorio de datos preparados. |
| [requirements.txt](requirements.txt) | Lista de dependencias del proyecto original. |

## Endpoints presentes en el código

| Ruta | Consulta |
| --- | --- |
| /peliculas_idioma/{Idioma} | Cantidad de películas por idioma. |
| /peliculas_duracion/{Pelicula} | Duración y año de lanzamiento. |
| /franquicia/{franquicia} | Cantidad, ingresos y promedio por franquicia. |
| /peliculas_pais/{pais} | Cantidad por país. |
| /productoras_exitosas/{productora} | Ingresos y cantidad de películas. |
| /get_director/{nombre_director} | Información e ingresos asociados a un director. |
| /recomendacion/{titulo} | Lista de películas similares según las características vectorizadas. |

## Reproducibilidad: advertencias conocidas

Actualmente el archivo **main.py** lee el dataset mediante una ruta absoluta de Windows, por lo que no puede ejecutarse directamente desde cualquier equipo. Además, el archivo **requirements.txt** contiene la entrada *ast*, que corresponde a un módulo de la biblioteca estándar de Python y debe revisarse.

Se detectó también una carpeta **venv** versionada en el repositorio. Es preferible excluir los entornos virtuales y reconstruirlos de forma local mediante un archivo de dependencias validado.

**No se han corregido esos problemas en esta propuesta**, porque esta ronda modifica únicamente documentación. La guía de ejecución reproducible deberá añadirse después de corregir el código y las dependencias.

## Limitaciones y próximos pasos

- Verificar la procedencia, licencia y esquema del dataset.
- Eliminar rutas absolutas de la aplicación y excluir el entorno virtual del control de versiones.
- Probar el arranque y los endpoints en un entorno limpio.
- Medir la calidad de recomendaciones con criterios explícitos en lugar de inferirla de la implementación.
- Incorporar ejemplos reales de respuestas y capturas de la API.

---

**Enfoque:** ingeniería de datos, procesamiento de texto y sistemas de recomendación.
