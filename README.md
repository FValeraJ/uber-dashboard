# Dashboard de Análisis de Viajes

## Descripción

Este proyecto es un dashboard interactivo para analizar datos de viajes de servicios de movilidad urbana. La aplicación está construida con Dash y Plotly en Python, y proporciona visualizaciones en tiempo real de métricas clave como reservas, cancelaciones, métodos de pago, calificaciones de conductores y distancias de viaje.

## Características

- **Visualizaciones Interactivas**: Gráficos que se actualizan en tiempo real basados en filtros de selección.
- **Múltiples Métricas**: KPIs para reservas totales, completadas, tasa de cancelación, precio promedio, distancia promedio y calificaciones.
- **Filtros Avanzados**: Selección por tipo de vehículo (Car, Auto, Bike) y rango de fechas.
- **Datos Adaptables**: Funciona con un dataset real o genera datos sintéticos para demostración.
- **Diseño Responsivo**: Interfaz optimizada con Bootstrap para diferentes dispositivos.
- **Análisis de Tendencia**: Incluye línea de regresión lineal para identificar patrones temporales.

## Tecnologías Utilizadas

- Python 3
- Dash
- Plotly
- Pandas
- NumPy
- Scikit-learn
- Dash Bootstrap Components

## Instalación

1. Clona este repositorio o descarga los archivos.
2. Navega al directorio del proyecto.
3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

1. Asegúrate de que el archivo de datos `ncr_ride_bookings.csv` esté en la carpeta `data/` (opcional, ya que la aplicación puede generar datos sintéticos).
2. Ejecuta la aplicación:

```bash
python app.py
```

3. Abre tu navegador y ve a `http://127.0.0.1:8050`.

## Estructura de Archivos

```
uber-dashboard/
├── ncr_ride_bookings.csv  # Dataset principal (opcional)
├── .gitignore                 # Archivos ignorados por Git
├── LICENSE                    # Licencia del proyecto
├── README.md                  # Este archivo
├── app.py                     # Aplicación principal Dash
├── requirements.txt           # Dependencias de Python
└── ncr_ride_bookings.csv      # Dataset (también en raíz por compatibilidad)
```

## Personalización

El dashboard está diseñado con una paleta de colores profesional que sigue la identidad de Uber. Los colores y estilos pueden modificarse en el código en la sección `COLORS`.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Autor

Desarrollado como parte de un portafolio profesional de análisis de datos y desarrollo de aplicaciones.
