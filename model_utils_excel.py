# model_utils_excel.py
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Carpeta donde están los excels
DATA_FOLDER = 'data/excel_cargado'


def cargar_excel():
    archivos = sorted(os.listdir(DATA_FOLDER))
    for archivo in archivos:
        if archivo.endswith('.csv'):
            print("Ruta de archivo csv", os.path.join(DATA_FOLDER, archivo))
            return pd.read_csv(os.path.join(DATA_FOLDER, archivo))

        elif archivo.endswith('.xlsx') or archivo.endswith('.xls'):
            print("Ruta de archivo xlsx", os.path.join(DATA_FOLDER, archivo))
            return pd.read_excel(os.path.join(DATA_FOLDER, archivo))
    raise FileNotFoundError("No se encontró un archivo válido")


def codificar_columnas(df):
    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    return df, le_dict


def graficar_modelo(X, y, modelo, nombre):
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue')
    if hasattr(modelo, 'predict'):
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = modelo.predict(x_line)
        ax.plot(x_line, y_pred, color='red')
    ax.set_title(nombre)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


def entrenar_modelos_desde_lista(combinaciones):
    df = cargar_excel()
    print("Columnas de archivo", list(df.columns.values))
    print("Total de registros en archivo:", df.count())
    df.dropna(inplace=False)
    df, codificadores = codificar_columnas(df)

    resultados = []

    for target, features in combinaciones:
        if isinstance(features, str):
            features = [features]

        try:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            resultado = {"target": target, "features": features, "modelos": []}

            # Regresión Lineal
            modelo_rl = LinearRegression()
            modelo_rl.fit(X_train, y_train)
            score_rl = modelo_rl.score(X_test, y_test)
            grafica_rl = None
            if len(features) == 1:
                grafica_rl = graficar_modelo(X[features[0]].values.reshape(-1, 1), y, modelo_rl, f"Regresión Lineal: {target}~{features[0]}")
            resultado["modelos"].append({"tipo": "regresion_lineal", "score": score_rl, "grafica": grafica_rl})

            # Regresión Logística 
            if len(set(y)) <= 10:
                modelo_log = LogisticRegression(max_iter=1000)
                modelo_log.fit(X_train, y_train)
                score_log = modelo_log.score(X_test, y_test)
                resultado["modelos"].append({"tipo": "regresion_logistica", "score": score_log})

            # Árbol de decisión
            modelo_arbol = DecisionTreeClassifier()
            modelo_arbol.fit(X_train, y_train)
            score_arbol = modelo_arbol.score(X_test, y_test)
            resultado["modelos"].append({"tipo": "arbol_decision", "score": score_arbol})

            # KMeans clustering
            if len(set(y)) > 2:  # solo si el target tiene más de 2 clases
                modelo_kmeans = KMeans(n_clusters=3, n_init=10)
                modelo_kmeans.fit(X)
                resultado["modelos"].append({"tipo": "clustering", "labels": modelo_kmeans.labels_.tolist()})

            resultados.append(resultado)
        except Exception as e:
            resultados.append({"target": target, "features": features, "error": str(e)})

    return resultados


#Se aplica RandomForest para generar un set de datos sintéticos para alimentar el modelo de regresión lineal.
def generar_datos_con_rf_col_adic(n=1000):
    np.random.seed(42)
    
    #Establecimiento de rango para valores aleatorios.
    #Datos dentro de parámentros son según el estimado para cada columna
    avg_daily_usage_hours = np.random.randint(1, 10, size=n)  # 1-10 horas
    sleep_hours_per_night = np.random.randint(3, 10, size=n)   # 3-10 horas
    
    # Ajuste para establecer a partir de qué valores podría aumentar datos en columna objetivo.
    base_score = np.where(avg_daily_usage_hours > 2, 
                         np.interp(avg_daily_usage_hours, [2, 12], [5, 7]), 
                         1)
    
    sleep_adjustment = np.where(sleep_hours_per_night < 6,
                              np.interp(sleep_hours_per_night, [3, 6], [2, 0]),
                              0)
    
    addicted_score = np.clip(base_score + sleep_adjustment, 1, 10)
    
    # Variabilidad para emular aleatoriedad de datos datos.
    addicted_score += np.random.normal(0, 0.5, size=n)
    addicted_score = np.clip(addicted_score, 1, 10).round().astype(int)
    
    return pd.DataFrame({
        'avg_daily_usage_hours': avg_daily_usage_hours,
        'sleep_hours_per_night': sleep_hours_per_night,
        'addicted_score': addicted_score
    })

def generar_datos_booleanos_col_academic_perfomance(n=5000):
    np.random.seed(42)
    
    # Generar las mismas features base
    avg_daily_usage_hours = np.random.randint(0, 12, size=n)
    sleep_hours_per_night = np.random.randint(3, 10, size=n)
    
    # Lógica para determinar si afecta el rendimiento
    prob_afecta = np.where(
        avg_daily_usage_hours > 3,
        np.interp(avg_daily_usage_hours, [3, 12], [0.3, 0.9]),  # Más horas → mayor probabilidad
        0.1  # Probabilidad base si usa poco
    )
    
    # Ajustar por horas de sueño
    prob_afecta += np.where(
        sleep_hours_per_night < 6,
        np.interp(sleep_hours_per_night, [3, 6], [0.4, 0]),  # Menos sueño → mayor probabilidad
        0
    )
    
    # Generar valores booleanos (1 o 0) basados en la probabilidad
    affects_academic_performance = np.random.binomial(1, np.clip(prob_afecta, 0, 1))
    
    return pd.DataFrame({
        'avg_daily_usage_hours': avg_daily_usage_hours,
        'sleep_hours_per_night': sleep_hours_per_night,
        'affects_academic_performance': affects_academic_performance
    })


#Función de predicción columna adict v.0.1
def prediccion_col_adiccion(df):
   
    # Paso 1: Generar datos sintéticos
    df_sintetico = generar_datos_con_rf_col_adic(5000)
    
    # Paso 2: Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df_sintetico[['avg_daily_usage_hours', 'sleep_hours_per_night']], 
              df_sintetico['addicted_score'])
    
    # Paso 3: Predecir valores faltantes
    mask = df['addicted_score'] == 0
    if mask.any():
        X = df.loc[mask, ['avg_daily_usage_hours', 'sleep_hours_per_night']]
        df.loc[mask, 'addicted_score'] = model.predict(X).round().clip(1, 10)
    
    return df

def prediccion_col_affect_academic_performance(df):
    # 1. Generar datos sintéticos
    df_sintetico = generar_datos_booleanos_col_academic_perfomance(5000)
    
    # 2. Entrenar modelo de clasificación (no regresión)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(
        df_sintetico[['avg_daily_usage_hours', 'sleep_hours_per_night']],
        df_sintetico['affects_academic_performance']
    )
    
    # 3. Predecir valores faltantes (asumiendo que los faltantes son 0)
    mask = df['affects_academic_performance'] == 0
    if mask.any():
        X = df.loc[mask, ['avg_daily_usage_hours', 'sleep_hours_per_night']]
        df.loc[mask, 'affects_academic_performance'] = model.predict(X)
    
    return df


# def prediccion_col_adiccion(df, guardar_cambios=False, ruta_guardado=None):
   
#     # 1. Generar datos sintéticos
#     df_sintetico = generar_datos_con_rf_col_adic(5000)
    
#     # 2. Entrenar modelo
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(df_sintetico[['avg_daily_usage_hours', 'sleep_hours_per_night']], 
#               df_sintetico['addicted_score'])
    
#     # 3. Identificar registros a modificar
#     mascara_ceros = df['addicted_score'] == 0
#     registros_a_modificar = mascara_ceros.sum()
    
#     if registros_a_modificar > 0:
#         # 4. Realizar predicciones
#         X = df.loc[mascara_ceros, ['avg_daily_usage_hours', 'sleep_hours_per_night']]
#         predicciones = model.predict(X).round().clip(1, 10)
        
#         # 5. Insertar los valores en el DataFrame original
#         df.loc[mascara_ceros, 'addicted_score'] = predicciones
        
#         # 6. Opcional: Guardar cambios
#         if guardar_cambios:
#             if not ruta_guardado:
#                 raise ValueError("Debe proporcionar ruta_guardado cuando guardar_cambios=True")
#             df.to_excel(ruta_guardado, index=False)
    
#     # 7. Mostrar resumen de cambios
#     print(f"\nResumen de cambios:")
#     print(f"Registros totales: {len(df)}")
#     print(f"Registros modificados: {registros_a_modificar}")
#     print("\nEjemplo de registros modificados:")
#     print(df[mascara_ceros].head(5))
    
#     return df

##Bucle para evaluación de columnas a predecir y rellenado de estas.
##En elaboración. Alex:
def prediccion_columnas_vacias():
    #Guardamos las columnas a evaluar dentro de variables para mejor legibilidad.
    df=cargar_excel()
    test_lectura= df.head()
    print("Head de dataset cargado:", test_lectura)
    col1= df['addicted_score']
    col2= df['mental_health_score']
    col3= df['affects_academic_performance']
    #Variables de columnas para predicciones.
    col4= df['avg_daily_usage_hours']
    col5= df['sleep_hours_per_night']
    cols4and5= df[['avg_daily_usage_hours','sleep_hours_per_night']]    

    #test de función para predicción
    # def predict_col1_linear():
                #x=[[col4, col5]]
    x= df[['avg_daily_usage_hours'],['sleep_hours_per_night']]  
    y= df['addicted_score']

    # formatoRespuesta=[]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    modelForCol1 = LinearRegression()
    modelForCol1.fit(X_train, y_train)

    col1_predicted_values=modelForCol1.predict(X_test).tolist()
    print("Valores de predicciones para nivel de adicción:", col1_predicted_values)

    return col1_predicted_values

    #Inicializamos un loop para evaluar si las columnas especificadas poseen valores nulos o ceros.
    # while((col1 or col2 or col3) == None or 0):

    #     if(col1 == None or 0):
    #         def predict_col1_linear():
    #             #x=[[col4, col5]]
    #             x=cols4and5
    #             y=col1

    #             X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #             modelForCol1 = LinearRegression()
    #             modelForCol1.fit(X_train, y_train)

    #             col1_predicted_values=modelForCol1.predict(X_test)
    #             print("Valores de predicciones para nivel de adicción:", col1_predicted_values)

    #             return col1_predicted_values

                

                
                