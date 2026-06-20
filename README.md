# End-to-End Customer Churn Pipeline: De Big Data a MLOps y Business Intelligence

Este proyecto implementa una solución empresarial completa y transversal para predecir y prevenir la fuga de clientes (*Customer Churn*). La arquitectura abarca desde la ingesta distribuda de datos masivos utilizando la **Arquitectura Medallón (Lakehouse)**, el entrenamiento de un modelo predictivo optimizado, analítica avanzada de interpretabilidad, hasta el despliegue en producción con prácticas **MLOps** y un dashboard interactivo enfocado en el impacto financiero.

## Impacto de Negocio (ROI)
* **Riesgo Financiero Identificado:** El modelo detectó un impacto crítico de **$230.58K USD mensuales** en riesgo de fuga (aproximadamente **$2.7M USD anuales**).
* **Focalización Estratégica:** Gracias al análisis predictivo, se determinó que el **98.08%** del riesgo financiero está concentrado exclusivamente en clientes con contratos *Month-to-month* (Mes a mes), permitiendo al equipo de Marketing optimizar el presupuesto de retención al 100%.

---

## Stack Tecnológico
* **Data Engineering:** Databricks (Community Edition), PySpark, Spark SQL, Arquitectura Medallón (Bronce y Plata).
* **Machine Learning & AI:** Python, XGBoost, Scikit-Learn, SHAP (Explainable AI), GridSearchCV.
* **MLOps & Despliegue:** MLflow (Tracking & Artifacts Registry), Joblib para serialización.
* **Business Intelligence:** Power BI Desktop (Dashboarding Interactivo & Storytelling Financiero).

---

## Arquitectura del Proyecto

### 1. Ingeniería de Datos (Capa Medallón)
Utilizando **PySpark** para procesamiento distribuido, se estructuró el flujo en dos capas dentro de Databricks:
* **Capa Bronce:** Ingesta de los datos crudos en formato crudo manteniendo el histórico. Tratamiento inicial de esquemas.
* **Capa Plata (`plata_churn_master`):** Limpieza exhaustiva de datos, tipado estricto de variables, imputación avanzada de valores nulos en cargos totales y estandarización para su consumo por el modelo de IA.

### 2. Machine Learning & Optimización
* **Algoritmo Ganador:** Se entrenó un clasificador **XGBoost** diseñado para mitigar el fuerte desbalanceo de clases en la deserción.
* **Ajuste de Hiperparámetros:** Mediante `GridSearchCV` con Validación Cruzada (Cross-Validation), se encontró una configuración robusta y conservadora para evitar el sobreajuste (*overfitting*):
  * `learning_rate`: 0.01
  * `max_depth`: 3
  * `n_estimators`: 100
* **Métrica Estrella:** Se priorizó maximizar el **Recall** para asegurar capturar la mayor cantidad de clientes en riesgo real de fuga.

### 3. Inteligencia Artificial Explicable (SHAP)
Para romper el paradigma de la "caja negra" de la IA, se integraron valores **SHAP** (*SHapley Additive exPlanations*). Esto permitió demostrar científicamente al negocio que:
1. El tipo de contrato (*Contract Type*) es la variable con mayor peso predictivo.
2. Poseer un contrato a dos años actúa matemáticamente como un escudo protector definitivo contra el Churn.
<img width="792" height="940" alt="463b082e-f682-4809-8e17-74001cba6a34" src="https://github.com/user-attachments/assets/a7d5e7cb-0259-43bc-b9a3-bbafaf263292" />


### 4. Automatización y MLOps
* **Gobierno de Modelos:** Integración con **MLflow Tracking** para registrar cada experimento, hiperparámetros y métricas de rendimiento en producción.
* **Pipeline de Inferencia:** Creación de una tubería de producción con serialización local (`joblib`) y una función automatizada que simula la llegada de clientes en tiempo real. Utiliza la técnica de `.reindex()` de Pandas para evitar rupturas de código si los datos entrantes omiten variables de entrenamiento.
<img width="1142" height="622" alt="7a524da5-27f3-463f-ac74-7d67e78ae5f1" src="https://github.com/user-attachments/assets/9acedbdb-6e0e-4c6b-8f6b-701526722245" />


### 5. Visualización Directiva (Power BI)
Se exportaron las predicciones completas enriquecidas con dos nuevas métricas generadas por el modelo: `Prediccion_Churn` y `Probabilidad_Fuga`. El reporte final automatiza:
* **El Hero KPI:** Cuantificación monetaria instantánea del Churn.
* **El Radar de Acción:** Una lista dinámica y priorizada ordenada por nivel de certeza (encabezada por alertas de clientes críticos con riesgos de fuga superiores al 60-70%) para llamadas de retención inmediatas.
<img width="790" height="492" alt="dashboard" src="https://github.com/user-attachments/assets/dea8af06-373b-4b0d-ab04-61b6f19b38e2" />


---

## Ejemplo del Pipeline de Inferencia (Código en Producción)

```python
import joblib
import pandas as pd

# Carga del modelo versionado
modelo_produccion = joblib.load("mejor_modelo_churn_xgb.pkl")
columnas_requeridas = joblib.load("columnas_modelo_churn.pkl")

def predecir_riesgo_cliente(datos_diccionario):
    df_nuevo = pd.DataFrame([datos_diccionario])
    df_encoded = pd.get_dummies(df_nuevo)
    # Truco Senior: Alineación forzada ante variables ausentes
    df_listo = df_encoded.reindex(columns=columnas_requeridas, fill_value=0)
    
    prediccion = modelo_produccion.predict(df_listo)[0]
    probabilidad = (modelo_produccion.predict_proba(df_listo)[0][1] * 100).round(2)
    
    estado = "¡ALERTA DE FUGA!" if prediccion == 1 else "Cliente Seguro"
    return f"Resultado: {estado} | Probabilidad de cancelación: {probabilidad}%"
