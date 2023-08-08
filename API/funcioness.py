from fastapi import FastAPI
import pandas as pd
import numpy as np
import json
import ast


#iniciamos FastAPI y el dataframe
app = FastAPI()
data = pd.read_csv("G:\Mi unidad\SOYHENRY\CIENCIA DE DATOS\CURSO\PROYECTOS\PI1_ML\datasetLimpio\datasetLimpio.csv")


#definimos la función que transformará en lista todos aquellos datos que perdieron esta condición
def obtener(celda):
    if pd.isnull(celda):
        return None
    if isinstance(celda, str) and celda.startswith("[") and celda.endswith("]"):
        try:
            return ast.literal_eval(celda)
        except (ValueError, SyntaxError):
            return celda  # Devuelve la celda original si no se puede convertir a lista
    return celda


#aplicamos esta función a todo el dataframe
data = data.applymap(obtener)


#definimos aquella función que al ingresar el idioma, retorna la cantidad de peliculas producidas en el mismo
@app.get("/peliculas_idioma/{Idioma}")
def peliculas_idioma(Idioma: str):

    cantidad = len(data[data["original_language"]==Idioma])
    
    return {"Idioma": Idioma, "cantidad": cantidad}


#definimos aquella función que al ingresar la pelicula, retorna la duración y el año
@app.get("/peliculas_duracion/{Pelicula}")
def peliculas_duracion(Pelicula: str):
    
    duracion = data.loc[data["title"] == Pelicula, "runtime"].iloc[0]

    anio = data.loc[data["title"] == Pelicula, "release_year"].iloc[0]

    return {'pelicula': Pelicula, 'duracion':duracion, 'anio':anio}


#definimos aquella función que al ingresar la franquicia, retorna la cantidad de películas, ganancia total y promedio
@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):

    cantidad = len(data[data["nameBTC"]==franquicia])

    ganancia_promedio = data[data["nameBTC"]==franquicia]["revenue"].mean()

    ganancia_total = data[data["nameBTC"]==franquicia]["revenue"].sum()

    return {'franquicia':franquicia, 'cantidad':cantidad, 'ganancia_total':ganancia_total, 'ganancia_promedio':ganancia_promedio}


#definimos aquella función que al ingresar el pais, retorna la cantidad de peliculas producidas en el mismo
@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais: str):

    cantidad = data["pcountry"].apply(lambda lista: pais in lista if isinstance(lista, list) else False).sum()
    
    return {'pais': pais, 'cantidad': cantidad}


#definimos aquella función que al ingresar la productora, entrega el revunue total y la cantidad de peliculas que realizó
@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''Ingresas la productora, entregandote el revunue total y la cantidad de peliculas que realizo '''

    revenuet = data[data["name_companie"].apply(lambda x: isinstance(x, list) and productora in x)]["revenue"].sum()

    cantidadt = data[data["name_companie"].apply(lambda x: isinstance(x, list) and productora in x)]["name_companie"].count()
    
    return {'productora':productora, 'revenue_total': revenuet,'cantidad':cantidadt}


#definimos aquella funcion que al ingresar el nombre de un director que se encuentre dentro de un dataset debe
# devolver el éxito del mismo medido a través del retorno. 
#Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma. En formato lista
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):

    retorno_total_director = data[data["Director"]== nombre_director]["return"].sum()

    peliculas = data[data["Director"]== nombre_director]["title"].to_list()

    anio = data[data["Director"]== nombre_director]["release_year"].to_list()

    retorno_pelicula = data[data["Director"]== nombre_director]["return"].to_list()

    budget_pelicula = data[data["Director"]== nombre_director]["budget"].to_list()

    revenue_pelicula = data[data["Director"]== nombre_director]["revenue"].to_list()

    return {'director':nombre_director, 'retorno_total_director':retorno_total_director, 'peliculas':peliculas, 'anio':anio,
     'retorno_pelicula':retorno_pelicula, 'budget_pelicula':budget_pelicula, 'revenue_pelicula':revenue_pelicula}
