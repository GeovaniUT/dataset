# Plan de Implementación – Predicciones Relevantes para el Usuario  
Versión: 2.0 – Autor: IA o3  
Fecha: ⬚⬚⬚

---

## 0. Contexto

El formulario inicial aporta las variables:  
Age, Gender, AcademicLevel, Country, AvgDailyUsageHours, MostUsedPlatform, SleepHoursPerNight, RelationshipStatus.  
Con estos datos se generan:  
AffectsAcademicPerformance, MentalHealthScore, AddictedScore.

A partir de las 11 columnas resultantes diseñamos seis nuevas predicciones de alto valor para el usuario final.

---

## 1. Nuevas Predicciones

| # | Objetivo (Nueva Columna) | Tipo | Algoritmo | Variables Principales |
|---|--------------------------|------|-----------|-----------------------|
| 1 | SleepQualityCategory | Clasificación (Buena/Mala) | Árbol de Decisión | SleepHoursPerNight, MentalHealthScore, AddictedScore |
| 2 | ConflictRisk | Clasificación (Alta/Baja) | RandomForest | AddictedScore, RelationshipStatus, AvgDailyUsageHours, MostUsedPlatform |
| 3 | RecommendedScreenTime | Regresión (Horas) | Regresión Lineal | Age, AddictedScore, MentalHealthScore |
| 4 | SocialWellbeingScore | Regresión (1–10) | RandomForestRegressor | RelationshipStatus, MentalHealthScore, MostUsedPlatform, AddictedScore |
| 5 | StudyEfficiencyScore | Regresión (0–100) | GradientBoostingRegressor | AffectsAcademicPerformance, AvgDailyUsageHours, SleepHoursPerNight |
| 6 | HighAddictionRisk | Clasificación (Sí/No) | Regresión Logística | AddictedScore, AvgDailyUsageHours, SleepHoursPerNight |

---

## 2. Detalle por Predicción

### 1. SleepQualityCategory
- Buena si SleepHoursPerNight ≥ 7, Mala en caso contrario (etiqueta derivada).  
- Modelo: `DecisionTreeClassifier(max_depth=4)` para interpretabilidad.  
- Pasos:  
  1. Helper `SleepQualityDataHelper.py` – genera datos balanceados.  
  2. Predictor `SleepQuality.py` en `predictors/`.  
  3. Endpoint `GET /api/completar-sleep-quality`.

### 2. ConflictRisk
- Alta si ConflictsOverSocialMedia ≥ 3.  
- Variables: AddictedScore, RelationshipStatus, AvgDailyUsageHours, MostUsedPlatform.  
- Modelo: `RandomForestClassifier(n_estimators=120)`.

### 3. RecommendedScreenTime
- Predice horas de uso “sanas” para el usuario.  
- Modelo base: `LinearRegression`.  
- Salida redondeada a 0.5 h.

### 4. SocialWellbeingScore
- Escala 1–10 de bienestar social percibido.  
- Modelo: `RandomForestRegressor`.  
- Permite feedback al usuario sobre su vida social digital.

### 5. StudyEfficiencyScore
- Puntaje 0–100 que estima eficiencia de estudio.  
- Influida por AffectsAcademicPerformance, uso de redes y sueño.  
- Modelo: `GradientBoostingRegressor`.

### 6. HighAddictionRisk
- Etiqueta binaria: Sí si AddictedScore ≥ 7, No en caso contrario.  
- Modelo: `LogisticRegression(max_iter=1000, class_weight='balanced')` para calibrar probabilidades.

---

## 3. Integración en el Proyecto

1. Crear un helper por objetivo en `utils/data_helpers/`, siguiendo el patrón actual.  
2. Añadir los nuevos archivos predictor en `predictors/` con nombres en PascalCase.  
3. Ampliar `predictors/prediccion_cols_vacias.py` al siguiente flujo:  
   - AffectsAcademicPerformance (existente)  
   - AddictedScore (existente)  
   - MentalHealthScore (existente)  
   - SleepQualityCategory *(nuevo)*  
   - RecommendedScreenTime *(nuevo)*  
   - HighAddictionRisk *(nuevo)*  
   - ConflictRisk *(nuevo, depende de ConflictsOverSocialMedia si existe o se predice)*  
   - SocialWellbeingScore *(nuevo)*  
   - StudyEfficiencyScore *(nuevo)*  
4. Añadir endpoints REST en `routes/prediccion_cols_faltantes_routes.py` y, de ser necesario, gráficas en `graficas_routes.py`.  
5. Guardar modelos y codificadores en `modelos/<Objetivo>/` usando la utilidad `guardar_modelo`.  
6. Incluir combinaciones relevantes en `routes/ml_routes.py` para entrenamiento masivo y evaluación.

---

## 4. Métricas Recomendadas

- Clasificación: Accuracy, F1, AUC.  
- Regresión: MAE, RMSE, R2.  
- Incluir estas métricas en la respuesta JSON de cada endpoint para trazabilidad.

---

## 5. Roadmap

| Prioridad | Acción | Archivos |
|-----------|--------|----------|
| Alta | Implementar SleepQualityCategory y HighAddictionRisk | Helpers + Predictors + Rutas |
| Media | ConflictRisk y RecommendedScreenTime | Id. |
| Baja | SocialWellbeingScore y StudyEfficiencyScore | Id. |

---

## 6. Especificación Técnica Detallada

### 6.1 Estructura de Carpetas y Archivos Nuevos

```
predictors/
  ├─ SleepQuality.py
  ├─ ConflictRisk.py
  ├─ RecommendedScreenTime.py
  ├─ SocialWellbeingScore.py
  ├─ StudyEfficiencyScore.py
  └─ HighAddictionRisk.py

utils/data_helpers/
  ├─ SleepQualityDataHelper.py
  ├─ ConflictRiskDataHelper.py
  ├─ RecommendedScreenTimeDataHelper.py
  ├─ SocialWellbeingDataHelper.py
  ├─ StudyEfficiencyDataHelper.py
  └─ HighAddictionRiskDataHelper.py
```

### 6.2 Plantilla para Helpers

```python
# utils/data_helpers/SleepQualityDataHelper.py
import numpy as np
import pandas as pd

def GenerarDatosSleepQuality(N: int = 5000) -> pd.DataFrame:
    """
    Genera un DataFrame con columnas:
    SleepHoursPerNight, MentalHealthScore, AddictedScore, SleepQualityCategory
    """
    rng = np.random.default_rng(42)
    sleep = rng.integers(3, 10, N)
    mental = rng.integers(1, 11, N)
    addicted = rng.integers(1, 11, N)
    calidad = (sleep >= 7).astype(int)           # 1 = Buena, 0 = Mala
    return pd.DataFrame({
        "SleepHoursPerNight": sleep,
        "MentalHealthScore": mental,
        "AddictedScore": addicted,
        "SleepQualityCategory": calidad
    })
```

Regla: Todos los helpers siguen esta plantilla cambiando distribución y columna objetivo.

### 6.3 Plantilla para Predictors

```python
# predictors/SleepQuality.py
from utils.data_helpers.SleepQualityDataHelper import GenerarDatosSleepQuality
from sklearn.tree import DecisionTreeClassifier

def CompletarSleepQualityCategory(df):
    """
    Rellena la columna SleepQualityCategory (0/1) si está vacía.
    Devuelve DataFrame actualizado.
    """
    sintetico = GenerarDatosSleepQuality(6000)
    modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
    modelo.fit(
        sintetico[["SleepHoursPerNight", "MentalHealthScore", "AddictedScore"]],
        sintetico["SleepQualityCategory"]
    )

    mask = df["SleepQualityCategory"].isna()
    if mask.any():
        X_pred = df.loc[mask, ["SleepHoursPerNight", "MentalHealthScore", "AddictedScore"]]
        df.loc[mask, "SleepQualityCategory"] = modelo.predict(X_pred)

    return df
```

Puntos comunes para los seis predictors:  
1. Importan su helper.  
2. Entrenan con 3–4 variables clave.  
3. Detectan valores vacíos (`isna()` o `== 0`).  
4. Devuelven el DataFrame enriquecido.

### 6.4 Definición de Endpoints

| Endpoint | Método | Parámetros URL | Descripción |
|----------|--------|----------------|-------------|
| `/api/completar-sleep-quality` | GET | – | Devuelve JSON con registros llenados, métricas y ejemplo de cambios. |
| `/api/completar-conflict-risk` | GET | – | Llena ConflictRisk.|
| … | … | … | Se replica el patrón para los otros cuatro objetivos. |

Formato de respuesta:

```json
{
  "totalRegistros": 250,
  "registrosModificados": 42,
  "metrics": { "accuracy": 0.88 },
  "ejemploCambios": {
      "antes": [{ /* … */ }],
      "despues": [{ /* … */ }]
  },
  "mensaje": "Se completaron 42 registros de SleepQualityCategory"
}
```

### 6.5 Actualización del Pipeline Global

1. **prediccion_cols_vacias.py**  
   Agregar en orden:

```python
from predictors.SleepQuality import CompletarSleepQualityCategory
from predictors.RecommendedScreenTime import CompletarRecommendedScreenTime
from predictors.HighAddictionRisk import CompletarHighAddictionRisk
from predictors.ConflictRisk import CompletarConflictRisk
from predictors.SocialWellbeingScore import CompletarSocialWellbeingScore
from predictors.StudyEfficiencyScore import CompletarStudyEfficiencyScore
```

2. Insertar cada llamada tras las existentes, respetando dependencias.

### 6.6 Inclusión en Entrenamiento Masivo

Añadir combinaciones en `routes/ml_routes.py`:

```python
combinaciones += [
    ("SleepQualityCategory", ["SleepHoursPerNight", "MentalHealthScore", "AddictedScore"]),
    ("ConflictRisk", ["AddictedScore", "RelationshipStatus", "AvgDailyUsageHours", "MostUsedPlatform"]),
    ("RecommendedScreenTime", ["Age", "AddictedScore", "MentalHealthScore"]),
    ("SocialWellbeingScore", ["RelationshipStatus", "MentalHealthScore", "MostUsedPlatform", "AddictedScore"]),
    ("StudyEfficiencyScore", ["AffectsAcademicPerformance", "AvgDailyUsageHours", "SleepHoursPerNight"]),
    ("HighAddictionRisk", ["AddictedScore", "AvgDailyUsageHours", "SleepHoursPerNight"])
]
```

### 6.7 Métricas y Persistencia

- Guardar cada modelo con `guardar_modelo`.  
- Persistir codificadores si aparecen variables categóricas nuevas.  
- Serializar métricas (Accuracy, MAE, etc.) en el JSON de respuesta.

### 6.8 Visualización y Gráficas

Cada predictor deberá entregar, además del DataFrame enriquecido, **una gráfica en base64** fácilmente interpretable en el front-end.  El patrón es idéntico al usado en `predictors/adiction_grafic.py`.

| Objetivo | Tipo de gráfica sugerida | Descripción rápida |
|----------|-------------------------|--------------------|
| SleepQualityCategory | Tarta (pie) 2 secciones | Proporción Buena/Mala con marcador “Tú”. |
| ConflictRisk | Barra apilada | Riesgo Alto vs Bajo; colorear barra del usuario. |
| RecommendedScreenTime | Scatter + línea óptima | Puntos históricos y línea objetivo, resaltar al usuario. |
| SocialWellbeingScore | Barra individual | Score (1-10) del usuario + promedio de la cohorte. |
| StudyEfficiencyScore | Gauge semicircular | Eficiencia 0-100 con zonas verde-amarillo-rojo. |
| HighAddictionRisk | Donut binario | Fracción de población en Riesgo Sí/No, parte central muestra tu %. |

#### Plantilla de helper para gráficas

```python
def PlotToBase64(fig):
    import base64, io, matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
```

Ejemplo de implementación en un predictor:

```python
# predictors/SleepQuality.py  (añadir al final de CompletarSleepQualityCategory)
import matplotlib.pyplot as plt
from .plot_utils import PlotToBase64   # utilidad común

def GenerarGraficaSleepQuality(df, usuario_idx):
    fig, ax = plt.subplots()
    counts = df["SleepQualityCategory"].value_counts()
    labels = ["Mala", "Buena"]
    ax.pie(counts, labels=labels, autopct="%1.0f%%", colors=["#e74c3c", "#2ecc71"])
    ax.set_title("Calidad de Sueño de la Comunidad")
    # Resaltar usuario
    categoria = df.loc[usuario_idx, "SleepQualityCategory"]
    ax.legend([f"Tú: {labels[categoria]}"], loc="upper right")
    return PlotToBase64(fig)
```

#### Extensión de Endpoints

| Endpoint | URL Ejemplo | Devuelve |
|----------|-------------|----------|
| `/api/grafica-sleep-quality/<int:userId>` | `/api/grafica-sleep-quality/15` | `{ "grafica": "data:image/png;base64,...", "detalle": {...} }` |
| `/api/grafica-conflict-risk/<int:userId>` | – | Ídem |
| … | … | … |

Las rutas se añaden a `routes/graficas_routes.py`, siguiendo la estructura de las ya existentes (`grafica-adiccion`, `grafica-regresion`, etc.).

### 6.9 Ajustes al Roadmap

| Prioridad | Acción |
|-----------|--------|
| Alta | Implementar util común `plot_utils.py` + gráficas SleepQuality y HighAddictionRisk |
| Media | Gráficas ConflictRisk y RecommendedScreenTime |
| Baja | Gráficas SocialWellbeingScore y StudyEfficiencyScore |

---

> Con esta ampliación, el plan pasa de nivel conceptual a guía práctica lista para ser implementada. Todas las variables y funciones respetan PascalCase y la descripción se mantiene en español.
