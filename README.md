# Proyecto Final · Machine Learning con PySpark y Docker
**Universidad Santo Tomás · Programa de Estadística · 2026-I**

---

## Estudiante
**Nombre completo:** Valentina Muñoz Palma  
**Repositorio:** [\[URL DEL REPOSITORIO\]](https://github.com/pinkandredval/proyecto_final_Munoz_Valentina.git)

---

## Descripción del problema

Este proyecto realiza un análisis 360° sobre dos fenómenos colombianos usando PySpark y Docker.

**Bloques 1 y 2 — Accidentalidad vial en Bogotá D.C.:**
El dataset de accidentalidad vial histórica de Bogotá contiene 199,146 registros de siniestros ocurridos entre 2015 y 2021, con información de gravedad, clase de accidente, localidad, coordenadas geográficas y franja horaria. El objetivo es entender los patrones temporales y geográficos de la accidentalidad, identificar grupos de riesgo mediante clustering, y predecir la gravedad del accidente (SOLO DANOS / CON HERIDOS / CON MUERTOS) a partir de variables contextuales.

**Bloque 3 — Reseñas de hoteles en español:**
El dataset contiene 19,993 reseñas de hoteles en español (tras limpieza: 11,536 reseñas únicas), etiquetadas con sentimiento positivo, negativo o regular. El objetivo es construir un modelo de análisis de sentimiento usando TF-IDF y compararlo con el modelo pre-entrenado de Hugging Face `pysentimiento/robertuito-sentiment-analysis`.

---

## Datasets utilizados

| Bloque | Dataset | Fuente |
|--------|---------|--------|
| 1 y 2 | Histórico de siniestros viales — Bogotá D.C. | [datos.gov.co — Secretaría Distrital de Movilidad](https://datosabiertos.bogota.gov.co/dataset/historico-siniestros-bogota-d-c) |
| 3 | Big_AHR.csv — Reseñas de hoteles en español | [Kaggle](https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced) |

---

## Estructura del repositorio

```
proyecto_final_[apellido_nombre]/
├── README.md
├── docker-compose.yml
├── bloque1_eda/
│   ├── bloque1_eda_munoz.ipynb
│   ├── datos_limpios.csv           
│   └── graficos/
├── bloque2_ml/
│   ├── bloque2_ml_munoz.ipynb
│   ├── graficos/
│   └── modelos/
│       └── modelo_randomforest/    
├── bloque3_nlp/
│   ├── bloque3_nlp_munoz.ipynb
│   └── graficos/
└── reporte_ejecutivo.pdf
```

---

## Cómo ejecutar los notebooks

### Requisitos previos

| Herramienta | Versión |
|-------------|---------|
| Python | 3.11.6 |
| PySpark | 3.5.0 |
| Docker | 29.2.1 |
| pysentimiento | 0.7.3 |
| torch | 2.12.0+cu130 |


### Dependencias principales

```bash
!pip install transformers pysentimiento torch -q
!pip install langdetect -q
```

### Orden de ejecución

> El Bloque 2 depende del archivo `datos_limpios.csv` generado por el Bloque 1.
> El Bloque 3 es completamente independiente.

1. **Bloque 1** — `bloque1_eda/bloque1_eda_munoz.ipynb`
   - Genera `bloque1_eda/datos_limpios.csv` al final

2. **Bloque 2** — `bloque2_ml/bloque2_ml_munoz.ipynb`
   - Requiere que `datos_limpios.csv` exista en `bloque1_eda/`

3. **Bloque 3** — `bloque3_nlp/bloque3_nlp_munoz.ipynb`
   - Independiente, puede correrse en cualquier orden

---

## Conclusión integrada

El análisis conjunto de los tres bloques revela patrones complementarios sobre fenómenos colombianos de distinta naturaleza. En accidentalidad vial, el EDA identificó que Kennedy y Engativá concentran el mayor volumen de siniestros y que los atropellos vespertinos tienen mayor riesgo para peatones; el clustering con K-Means (K=6) descubrió que el Clúster 5 —dominado por atropellos— concentra el 78.5% de los accidentes mortales del dataset, mientras que los modelos de clasificación (Random Forest, F1=0.6469) confirmaron que el tipo de accidente y la hora son las variables más predictivas de la gravedad, aunque el dataset carece de variables de comportamiento del conductor que permitirían mejorar significativamente la predicción. En análisis de sentimiento, tanto TF-IDF (F1=0.8208) como Hugging Face (F1=0.8098) alcanzaron rendimientos similares en reseñas de hoteles, pero el modelo pre-entrenado demostró mayor robustez ante negaciones y ambigüedad lingüística, lo que lo hace preferible para producción. Juntos, los tres bloques muestran que los datos abiertos colombianos tienen valor analítico real, pero requieren enriquecimiento con variables contextuales para alcanzar su potencial predictivo completo.

---

## Limitaciones

1. **Bloque 1 y 2:** El dataset de accidentalidad no incluye variables de comportamiento del conductor (velocidad, uso de cinturón, estado del conductor), que son determinantes primarios de la gravedad. Esto limita el F1 del modelo clasificador a ~0.65.

2. **Bloque 2:** La clase CON MUERTOS representa solo el ~2% del dataset. A pesar del balanceo por `weightCol`, el recall de esta clase sigue siendo bajo debido al desbalance severo y a la limitación informativa descrita arriba.

3. **Bloque 3:** Ninguno de los dos modelos de sentimiento detecta sarcasmo de forma confiable. Para un sistema en producción se requeriría un modelo especializado o reglas específicas para ironía en español.

---


