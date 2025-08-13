# Latam Demand Forecast

Repositorio de proyectos de predicción de demanda aérea y optimización de precios, enfocados en LATAM Airlines.

---

## Proyecto principal: Predicción de demanda y optimización de precios

**Descripción:**  
Pipeline completo de Revenue Management para LATAM AIRLINES CHILE que combina:  
- **ANN (Redes Neuronales Artificiales):** Predice la demanda de vuelos por ruta y mes.  
- **Q-learning:** Optimiza precios dinámicamente según la demanda y la ocupación.  

Incluye **carga y limpieza de datos**, **preprocesamiento**, entrenamiento de la ANN, simulación de decisiones de precios y generación de **estimaciones de ingresos y precios finales**. Basado en un caso práctico para el cargo de Analista Revenue Management.

**Tecnologías:**  
Python 3.10, Pandas, NumPy, Scikit-learn, TensorFlow (Keras), Matplotlib, Seaborn

**Datos:**  
Informes públicos de tráfico aéreo de la DGAC (Chile) para LATAM AIRLINES CHILE, año 2024.

**Estructura:**  
- `datos_latam_2024/` → Archivos Excel de DGAC.  
- `Modelo-Latam.py` → Pipeline de limpieza, ANN y Q-learning.  
- `resultados_latam_2024_simulacion.csv` → Resultados con demanda estimada, precios y ingresos.

---

## Uso

1. Clonar el repositorio.  
2. Crear un entorno virtual con Python 3.10.  
3. Instalar dependencias:  
   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
Ejecutar el script principal:

bash
Copiar
Editar
python Modelo-Latam.py
Consultar los resultados en resultados_latam_2024_simulacion.csv.

Próximos pasos
Extender modelos de aprendizaje por refuerzo para optimización de precios.

Incluir más fuentes de datos históricas.

Crear visualizaciones interactivas para análisis de resultados.

Contacto
Email: alejandrarobles2509@gmail.com
GitHub: https://github.com/Earobless
LinkedIn:www.linkedin.com/in/erika-alejandra-robles-sosa-082600262

Erika Alejandra Robles Sosa
Diplomado en Ciencia de Datos para las Finanzas - Universidad de Chile, 2025
