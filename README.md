# Dashboard de Análisis de Viajes - Portafolio Profesional

## Descripción del Proyecto

Este proyecto es un dashboard interactivo desarrollado en Python que permite analizar datos de movilidad urbana y servicios de transporte. La aplicación proporciona visualizaciones en tiempo real de métricas clave como reservas, cancelaciones, métodos de pago, calificaciones de conductores y distancias de viaje.

## Características Principales

- **Visualización Interactiva**: Gráficos dinámicos que se actualizan en tiempo real según los filtros seleccionados
- **Múltiples Métricas**: KPIs para reservas totales, completadas, tasa de cancelación, precio promedio, distancia promedio y calificaciones
- **Filtros Avanzados**: Selección por tipo de vehículo y rango de fechas
- **Datos Adaptables**: Funciona tanto con dataset original como con datos sintéticos generados automáticamente
- **Diseño Responsivo**: Interfaz optimizada que se adapta a diferentes dispositivos
- **Análisis de Tendencia**: Incluye línea de regresión para identificar patrones temporales

## Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal
- **Dash**: Framework para aplicaciones web analíticas
- **Plotly**: Biblioteca para visualizaciones interactivas
- **Pandas**: Manipulación y análisis de datos
- **Scikit-learn**: Algoritmos de machine learning (regresión lineal)
- **Bootstrap**: Framework CSS para diseño responsivo
- **Dash Bootstrap Components**: Componentes de Bootstrap para Dash

## Estructura del Proyecto

```
proyecto/
├── data/                 # Carpeta para datasets
│   └── ncr_ride_bookings.csv  # Dataset principal (opcional)
├── __pycache__/         # Archivos de caché de Python
├── app.py               # Aplicación principal Dash
├── requirements.txt     # Dependencias del proyecto
└── README.md           # Este archivo
```

## Instalación y Uso

1. Clona o descarga el proyecto
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Coloca tu dataset en la carpeta `data/` (opcional)
4. Ejecuta la aplicación:
   ```bash
   python app.py
   ```
5. Abre tu navegador en: http://127.0.0.1:8050

## Funcionalidades

### Panel de Control Principal
- Visualización de 6 KPIs principales en tarjetas informativas
- Filtros interactivos por tipo de vehículo y fechas
- Opción para mostrar/ocultar línea de tendencia

### Visualizaciones
- Gráfico circular de estados de reserva
- Gráfico de barras de métodos de pago
- Diagrama de violín para distribución de calificaciones
- Histograma con boxplot para distancias de viaje
- Serie temporal con tendencia para reservas por fecha

### Tipos de Vehículos
- Automóviles (Car)
- Autos (Auto)
- Motocicletas (Bike)

## Personalización

El dashboard está diseñado con una paleta de colores profesional y accesible, manteniendo la identidad visual de Uber como referencia. Los colores y estilos pueden modificarse fácilmente en la sección de configuración de colores del código.

## Notas Técnicas

- La aplicación intenta cargar primero el dataset original desde múltiples ubicaciones posibles
- Si no encuentra el dataset, genera automáticamente datos sintéticos realistas para demostración
- Incluye manejo de errores y advertencias para mayor robustez
- Optimizado para rendimiento con agrupaciones eficientes de datos

## Propósito

Este proyecto fue desarrollado como demostración de habilidades en:
- Análisis de datos y visualización
- Desarrollo de aplicaciones interactivas con Dash
- Creación de dashboards profesionales
- Manipulación y generación de datasets
- Machine learning aplicado (regresión lineal para tendencias)

## Autor

Desarrollado como parte de un portafolio profesional de análisis de datos y desarrollo de aplicaciones.
