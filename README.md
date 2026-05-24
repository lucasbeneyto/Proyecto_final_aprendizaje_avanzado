# Predicción del abandono universitario

Este repositorio contiene el trabajo final de la asignatura **Aprendizaje Avanzado**, centrado en la predicción del abandono universitario mediante modelos de Machine Learning.

El objetivo principal del proyecto es construir un modelo capaz de identificar estudiantes con riesgo de abandono académico a partir de información académica, administrativa y socioeconómica. Además del rendimiento predictivo, también se analiza la explicabilidad del modelo y posibles diferencias de comportamiento entre distintos grupos de estudiantes.

## Descripción del problema

El abandono universitario es un problema relevante tanto para los estudiantes como para las instituciones educativas. Detectar de forma temprana a estudiantes en riesgo puede ayudar a activar medidas de apoyo, como tutorías, seguimiento académico o ayuda administrativa.

En este trabajo se plantea el problema como una tarea de clasificación binaria:

- `Dropout`: el estudiante abandona.
- `No Dropout`: el estudiante se gradúa o continúa matriculado.

La clase positiva del problema es `Dropout`, ya que es el caso que se quiere detectar.

## Dataset

El dataset utilizado contiene información de **4424 estudiantes** y **35 variables**. La variable objetivo original tiene tres clases:

- `Graduate`
- `Enrolled`
- `Dropout`

Para simplificar el problema y adaptarlo al objetivo del trabajo, se transforma en una variable binaria:

- `Graduate` + `Enrolled` → `No Dropout`
- `Dropout` → `Dropout`

El conjunto de datos incluye variables académicas, como asignaturas matriculadas, aprobadas y calificaciones; variables administrativas, como tasas al día o deuda; y variables socioeconómicas, como edad, beca, género o situación internacional.

## Estructura del repositorio

```text
Proyecto_final_aprendizaje_avanzado/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_modelos.ipynb
│   ├── 04_xai_shap.ipynb
│   └── 05_fairness.ipynb
│
├── results/
│   ├── figures/
│   ├── models/
│   └── tables/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── fairness.py
│
├── INFORME_GUIA.md
├── requirements.txt
└── README.md
```

## Notebooks

El proyecto se ha organizado en varios notebooks para separar cada parte del trabajo y que sea más fácil seguir el proceso completo.

### `01_eda.ipynb`

En este notebook se realiza el análisis exploratorio inicial del dataset. Se revisa el tamaño del conjunto de datos, la distribución de la variable objetivo y algunas relaciones entre el abandono y distintas variables académicas, administrativas y socioeconómicas.

### `02_baseline.ipynb`

Aquí se entrena el primer modelo de referencia. Se utiliza una regresión logística como baseline para tener un punto de comparación sencillo antes de probar modelos más complejos.

### `03_modelos.ipynb`

En este notebook se entrenan y comparan distintos modelos de clasificación:

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- Gradient Boosting

La comparación se hace usando métricas como accuracy, precision, recall, F1-score y ROC-AUC.

### `04_xai_shap.ipynb`

En este notebook se analiza la explicabilidad del modelo final usando SHAP. La idea es entender qué variables tienen más peso en las predicciones y evitar tratar el modelo como una caja negra.

### `05_fairness.ipynb`

Aquí se estudia si el modelo se comporta de forma diferente entre distintos grupos de estudiantes. Para ello se analizan variables como género, beca, deuda, tasas académicas al día e internacionalidad.

## Metodología

El flujo general del trabajo ha sido el siguiente:

1. Carga y revisión inicial del dataset.
2. Transformación de la variable objetivo en un problema binario.
3. Separación entre variables predictoras y variable objetivo.
4. División de los datos en entrenamiento y test.
5. Preprocesamiento de las variables:
   - escalado de variables numéricas;
   - codificación One-Hot de variables categóricas.
6. Entrenamiento de distintos modelos de clasificación.
7. Comparación de resultados.
8. Selección del modelo final.
9. Análisis de explicabilidad con SHAP.
10. Análisis de fairness por grupos.

También se eliminaron variables futuras, especialmente las relacionadas con el segundo semestre, para evitar problemas de *data leakage*. Esto es importante porque el objetivo del trabajo es simular una predicción temprana del abandono, usando información que estaría disponible en un contexto real.

## Modelo final

El modelo seleccionado como modelo final fue **Gradient Boosting**.

Aunque la regresión logística obtuvo resultados muy competitivos, se escogió Gradient Boosting porque mantiene un rendimiento alto y permite realizar un análisis de interpretabilidad más completo mediante SHAP.

Resultados principales del modelo final:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.86 |
| Precision | 0.83 |
| Recall | 0.71 |
| F1-score | 0.76 |
| ROC-AUC | 0.91 |

El modelo consigue diferenciar bastante bien entre estudiantes con y sin riesgo de abandono. Aun así, el recall muestra que todavía hay estudiantes que realmente abandonan y que el modelo no detecta, por lo que no debería usarse como una herramienta automática de decisión.

## Explicabilidad

Para interpretar el modelo se ha utilizado SHAP.

El análisis muestra que la variable más influyente es el número de asignaturas aprobadas en el primer semestre. Esto tiene sentido, ya que el rendimiento académico temprano suele estar muy relacionado con el abandono universitario.

Otras variables relevantes son:

- tasas académicas al día;
- curso;
- edad de matrícula;
- asignaturas matriculadas;
- nota del primer semestre;
- beca;
- deuda;
- género.

Este análisis ayuda a entender mejor el comportamiento del modelo y permite justificar de forma más clara por qué se realizan ciertas predicciones.

## Fairness

Además del rendimiento general, se estudió si el modelo se comporta de forma diferente entre distintos grupos de estudiantes.

Las variables analizadas fueron:

- `Gender`
- `Scholarship holder`
- `Debtor`
- `Tuition fees up to date`
- `International`

El análisis muestra que el rendimiento por género es relativamente equilibrado, pero aparecen diferencias más claras en variables relacionadas con la situación económica o administrativa, como beca, deuda o tasas académicas al día.

Esto es importante porque un modelo de este tipo podría reforzar desigualdades si se utiliza sin supervisión. Por ese motivo, el sistema debería usarse únicamente como apoyo para orientar intervenciones tempranas, no como una herramienta automática para etiquetar o penalizar estudiantes.

## Instalación

Para ejecutar el proyecto, se recomienda crear un entorno virtual e instalar las dependencias del archivo `requirements.txt`.

```bash
python -m venv venv
```

En Windows:

```bash
venv\Scripts\activate
```

En Linux o macOS:

```bash
source venv/bin/activate
```

Después, instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Los notebooks pueden ejecutarse en orden desde la carpeta `notebooks/`:

```text
01_eda.ipynb
02_baseline.ipynb
03_modelos.ipynb
04_xai_shap.ipynb
05_fairness.ipynb
```

También se incluyen scripts en la carpeta `src/` para reproducir partes principales del flujo de trabajo.

Entrenar y comparar modelos:

```bash
python src/train.py
```

Evaluar el modelo final:

```bash
python src/evaluate.py
```

Generar explicabilidad con SHAP:

```bash
python src/explain.py
```

## Resultados generados

Los resultados se guardan en la carpeta `results/`, organizada en:

```text
results/
├── figures/
├── models/
└── tables/
```

Entre los resultados principales se incluyen:

- comparación de modelos;
- matriz de confusión;
- curva ROC;
- curva Precision-Recall;
- importancia de variables con SHAP;
- métricas de fairness por grupo.

## Conclusión

El trabajo muestra que es posible construir un modelo con buen rendimiento para predecir el abandono universitario. El modelo final alcanza un ROC-AUC de 0.91 y un F1-score de 0.76, lo que indica una buena capacidad predictiva.

Sin embargo, el análisis también muestra que el modelo no debe utilizarse como una decisión automática. Algunas variables importantes están relacionadas con la situación económica o administrativa del estudiante, por lo que es necesario interpretar los resultados con cuidado.

La principal utilidad del sistema sería servir como herramienta de apoyo para detectar estudiantes en riesgo y priorizar medidas de acompañamiento. En ningún caso debería utilizarse para etiquetar, sancionar o excluir estudiantes sin revisión humana.

## Integrantes

- Lucas Benito
- Pablo Candela
- Alexander Herasimovich
