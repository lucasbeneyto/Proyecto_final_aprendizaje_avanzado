# Guía para el informe

## 1. Introducción y objetivo

- Presentar el problema: predicción del abandono estudiantil.
- Explicar por qué es relevante anticipar estudiantes en riesgo.
- Definir el objetivo técnico: construir, evaluar, explicar y auditar un modelo de clasificación.

## 2. Datos y preparación

- Describir el dataset: número de registros, tipos de variables y variable objetivo.
- Explicar la transformación de `Target` a `Target_bin`, donde `Dropout` es la clase positiva.
- Justificar las columnas descartadas para evitar fuga de información.
- Resumir el preprocesado: imputación, escalado numérico y one-hot encoding.

## 3. Análisis exploratorio

- Mostrar la distribución de la variable objetivo y comentar el posible desbalance.
- Analizar variables académicas, económicas y personales relevantes.
- Incluir el análisis de variables sensibles: género, beca, deuda, tasas al día e internacionalidad.
- Resumir patrones iniciales que sugieren mayor o menor riesgo de abandono.

## 4. Modelo baseline

- Explicar la regresión logística como punto de referencia.
- Presentar métricas de test y validación cruzada.
- Interpretar matriz de confusión, precisión, recall, F1 y ROC-AUC.
- Aclarar que en este problema los falsos negativos son especialmente importantes.

## 5. Comparación de modelos

- Enumerar los modelos probados: regresión logística, árbol, Random Forest, Gradient Boosting y SVM.
- Describir el ajuste de hiperparámetros con `GridSearchCV`.
- Comparar resultados de test y validación cruzada.
- Justificar la selección del mejor modelo final.

## 6. Resultados del modelo final

- Presentar las métricas principales del modelo elegido.
- Incluir matriz de confusión, curva Precision-Recall y curva ROC.
- Comentar fortalezas y debilidades del modelo según las métricas.

## 7. Explicabilidad con SHAP

- Explicar brevemente qué mide SHAP.
- Presentar importancia global de variables.
- Incluir los gráficos principales: barras, summary plot y beeswarm.
- Analizar casos locales: alto riesgo, bajo riesgo y caso dudoso.
- Conectar las variables importantes con interpretación educativa.

## 8. Análisis de fairness

- Definir las variables sensibles usadas.
- Presentar métricas por grupo: recall, F1, tasa de falsos positivos y tasa de falsos negativos.
- Destacar las mayores brechas entre grupos.
- Interpretar especialmente las brechas en falsos negativos, porque implican estudiantes en riesgo no detectados.

## 9. Mitigación

- Explicar la estrategia de umbrales por grupo.
- Comparar resultados antes y después de la mitigación.
- Mostrar el efecto sobre recall y tasa de falsos negativos.
- Reconocer el compromiso entre mejorar equidad y aumentar falsos positivos o reducir precisión.

## 10. Conclusiones y trabajo futuro

- Resumir el mejor modelo y sus resultados.
- Indicar qué variables parecen más influyentes en el abandono.
- Resumir los principales riesgos de fairness detectados.
- Proponer mejoras: más datos, validación temporal, variables adicionales, calibración, coste explícito de errores y revisión con expertos educativos.
