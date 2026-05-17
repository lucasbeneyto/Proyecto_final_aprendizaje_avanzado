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
├── requirements.txt
└── README.md
