import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ==================== CARGA DE DATASET ====================
import os

def generate_sample_data():
    """Genera datos de ejemplo realistas como fallback"""
    np.random.seed(42)  # Para resultados consistentes
    
    # Fechas del último año
    start_date = datetime.now() - timedelta(days=365)
    dates = pd.date_range(start_date, datetime.now(), freq='D')
    
    # Generar datos más realistas con patrones temporales
    sample_data = []
    for date in dates:
        # Más reservas en fines de semana y horas pico
        day_of_week = date.weekday()
        weekend_multiplier = 1.5 if day_of_week >= 5 else 1.0
        
        # Número de reservas por día (más realista)
        daily_bookings = int(np.random.poisson(30 * weekend_multiplier))
        
        for _ in range(daily_bookings):
            # Patrones más realistas por tipo de vehículo
            vehicle_probs = [0.6, 0.25, 0.15]  # Car, Auto, Bike
            vehicle_type = np.random.choice(['Car', 'Auto', 'Bike'], p=vehicle_probs)
            
            # Estado de reserva con tasas realistas
            status_probs = [0.85, 0.10, 0.05]  # Completed, Cancelled, Incomplete
            booking_status = np.random.choice(['Completed', 'Cancelled', 'Incomplete'], p=status_probs)
            
            # Métodos de pago
            payment_probs = [0.45, 0.35, 0.20]  # Wallet, Credit Card, Cash
            payment_method = np.random.choice(['Wallet', 'Credit Card', 'Cash'], p=payment_probs)
            
            # Valores más realistas por tipo de vehículo
            base_values = {'Car': 150, 'Auto': 80, 'Bike': 50}
            booking_value = np.random.normal(base_values[vehicle_type], 30)
            booking_value = max(20, booking_value)  # Mínimo realista
            
            # Distancias más realistas
            base_distances = {'Car': 12, 'Auto': 8, 'Bike': 5}
            ride_distance = np.random.exponential(base_distances[vehicle_type])
            ride_distance = min(50, max(1, ride_distance))  # Límites realistas
            
            # Calificaciones con distribución realista (sesgada hacia valores altos)
            rating = np.random.beta(8, 2) * 4 + 1  # Sesgada hacia calificaciones altas
            
            sample_data.append({
                'Date': date,
                'Vehicle Type': vehicle_type,
                'Booking Status': booking_status,
                'Payment Method': payment_method,
                'Booking Value': round(booking_value, 2),
                'Ride Distance': round(ride_distance, 2),
                'Driver Ratings': round(rating, 1)
            })
    
    return pd.DataFrame(sample_data)

def load_dataset():
    """Intenta cargar el dataset original, usa datos sintéticos como fallback"""
    
    # Rutas posibles para el dataset (puedes agregar más rutas aquí)
    possible_paths = [
        r"C:\Users\xenia\OneDrive\Desktop\CODE\uber-dashboard\data\ncr_ride_bookings.csv",
        "./data/ncr_ride_bookings.csv",
        "./ncr_ride_bookings.csv",
        "data/ncr_ride_bookings.csv",
        "ncr_ride_bookings.csv"
    ]
    
    # Intentar cargar desde cada ruta posible
    for dataset_path in possible_paths:
        if os.path.exists(dataset_path):
            try:
                print(f"📁 Intentando cargar dataset desde: {dataset_path}")
                df = pd.read_csv(dataset_path)
                
                # Limpiar datos como en el código original
                df.columns = df.columns.str.strip()
                for col in df.select_dtypes(include='object').columns:
                    df[col] = df[col].map(lambda x: x.strip('"') if isinstance(x, str) else x)
                
                print(f"✅ Dataset original cargado exitosamente: {len(df):,} registros")
                print(f"📊 Columnas disponibles: {list(df.columns)}")
                return df, True
                
            except Exception as e:
                print(f"❌ Error al cargar desde {dataset_path}: {e}")
                continue
    
    # Si no se pudo cargar el dataset original, usar datos sintéticos
    print("⚠️ No se encontró el dataset original en ninguna ubicación.")
    print("🔄 Generando datos sintéticos para demostración...")
    df = generate_sample_data()
    print(f"✅ Datos sintéticos generados: {len(df):,} registros")
    return df, False

# Cargar dataset (original o sintético)
df, using_real_data = load_dataset()

# Asegurar que las columnas estén en el formato correcto
df['Date'] = pd.to_datetime(df['Date'])
df['Is_Cancelled'] = df['Booking Status'].str.contains('Cancelled|Incomplete', case=False, na=False)

# ==================== CONFIGURACIÓN DE COLORES PROFESIONALES ====================
# Paleta de colores accesible y profesional
COLORS = {
    'primary': '#1FBAD6',      # Azul Uber (brand consistency)
    'success': '#00C853',      # Verde para métricas positivas
    'danger': '#FF5252',       # Rojo para métricas negativas  
    'warning': '#FF9800',      # Naranja para advertencias
    'info': '#9C27B0',         # Púrpura para información neutra
    'secondary': '#6C757D',    # Gris para elementos secundarios
    'light': '#F8F9FA',        # Fondo claro
    'dark': '#343A40'          # Texto oscuro
}

# ==================== CONFIGURACIÓN DE LA APLICACIÓN ====================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard de Análisis de Viajes - Portafolio Profesional"

# ==================== LAYOUT MEJORADO ====================
app.layout = dbc.Container([
    # Header con información del proyecto
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🚖 Dashboard de Análisis de Viajes", 
                       className="text-center mb-2", 
                       style={'color': COLORS['primary'], 'fontWeight': 'bold'}),
                html.P("Análisis interactivo de datos de movilidad urbana", 
                      className="text-center text-muted mb-0"),
                html.Small(f"* {'Usando dataset original' if using_real_data else 'Dashboard desarrollado con datos sintéticos para fines de demostración'}", 
                          className="text-center d-block", 
                          style={'color': COLORS['success'] if using_real_data else COLORS['secondary'], 
                                'fontStyle': 'italic'})
            ])
        ], width=12)
    ], className="my-4"),
    
    # Panel de filtros y KPIs
    dbc.Row([
        # Filtros laterales
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("🔍 Filtros de Análisis", className="m-0", style={'color': COLORS['primary']})
                ]),
                dbc.CardBody([
                    # Filtro de tipo de vehículo
                    html.Label("Tipo de Vehículo:", className="font-weight-bold mb-2"),
                    dcc.Dropdown(
                        id="vehicle_filter",
                        options=[{"label": "🚗 " + v if v == "Car" else "🛺 " + v if v == "Auto" else "🏍️ " + v, 
                                "value": v} for v in df["Vehicle Type"].unique()],
                        value=df["Vehicle Type"].unique()[0],
                        clearable=False,
                        style={'marginBottom': '20px'}
                    ),
                    
                    # Filtro de fechas
                    html.Label("Rango de Fechas:", className="font-weight-bold mb-2"),
                    dcc.DatePickerRange(
                        id='date_filter',
                        min_date_allowed=df['Date'].min(),
                        max_date_allowed=df['Date'].max(),
                        start_date=df['Date'].max() - timedelta(days=30),
                        end_date=df['Date'].max(),
                        display_format='DD/MM/YYYY',
                        style={'marginBottom': '20px'}
                    ),
                    
                    # Control para línea de tendencia
                    html.Label("Opciones de Visualización:", className="font-weight-bold mb-2"),
                    dbc.Checklist(
                        id="trend_line_toggle",
                        options=[{"label": " Mostrar línea de tendencia", "value": "show_trend"}],
                        value=["show_trend"],
                        style={'marginBottom': '10px'}
                    )
                ])
            ], className="shadow-sm h-100")
        ], width=12, lg=4),
        
        # KPIs principales
        dbc.Col([
            html.Div(id="kpi_cards")
        ], width=12, lg=8)
    ], className="mb-4"),
    
    # Primera fila de gráficos
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📊 Estado de Reservas", className="m-0")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="status_chart", config={'displayModeBar': False})
                ])
            ], className="shadow-sm h-100")
        ], width=12, lg=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("💳 Formas de Pago", className="m-0")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="payment_chart", config={'displayModeBar': False})
                ])
            ], className="shadow-sm h-100")
        ], width=12, lg=6),
    ], className="mb-4"),
    
    # Segunda fila de gráficos
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("⭐ Calificaciones de Conductores", className="m-0")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="ratings_chart", config={'displayModeBar': False})
                ])
            ], className="shadow-sm h-100")
        ], width=12, lg=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📏 Distancias de Viaje", className="m-0")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="distance_chart", config={'displayModeBar': False})
                ])
            ], className="shadow-sm h-100")
        ], width=12, lg=6),
    ], className="mb-4"),
    
    # Gráfico de tendencia temporal (ancho completo)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📈 Reservas por Fecha", className="m-0")
                ]),
                dbc.CardBody([
                    dcc.Graph(id="trend_chart", config={'displayModeBar': False})
                ])
            ], className="shadow-sm")
        ], width=12)
    ], className="mb-4"),
    
    # Footer informativo
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.P([
                            "📊 ", html.Strong("Información del Dashboard"), 
                            " | Última actualización: ", datetime.now().strftime('%d/%m/%Y %H:%M')
                        ], className="mb-2"),
                        html.P([
                            "🔧 Desarrollado con Python, Dash y Plotly | ",
                            f"📁 Datos: {'Dataset original NCR Ride Bookings' if using_real_data else 'Muestra sintética'} de ", f"{len(df):,}", " registros"
                        ], className="mb-0", style={'fontSize': '14px', 'color': COLORS['secondary']})
                    ])
                ])
            ], className="bg-light")
        ], width=12)
    ])
], fluid=True, style={'backgroundColor': COLORS['light']})

# ==================== FUNCIONES AUXILIARES ====================
def calculate_trend_line(dates, values):
    """Calcula la línea de tendencia usando regresión lineal"""
    try:
        # Convertir fechas a números ordinales para la regresión
        X = np.array([d.toordinal() for d in dates]).reshape(-1, 1)
        y = np.array(values)
        
        # Ajustar modelo de regresión lineal
        model = LinearRegression()
        model.fit(X, y)
        
        # Predecir valores de tendencia
        trend_values = model.predict(X)
        
        return trend_values, model.score(X, y)  # También devolver R²
    except:
        return np.array(values), 0  # Fallback en caso de error

# ==================== CALLBACKS PRINCIPALES ====================
@app.callback(
    [Output("kpi_cards", "children"),
     Output("status_chart", "figure"),
     Output("payment_chart", "figure"),
     Output("ratings_chart", "figure"),
     Output("distance_chart", "figure"),
     Output("trend_chart", "figure")],
    [Input("vehicle_filter", "value"),
     Input("date_filter", "start_date"),
     Input("date_filter", "end_date"),
     Input("trend_line_toggle", "value")]
)
def update_dashboard(vehicle, start_date, end_date, trend_options):
    """Callback principal que actualiza todo el dashboard"""
    
    # Filtrar datos según selecciones del usuario
    dff = df[(df["Vehicle Type"] == vehicle) &
             (df["Date"] >= pd.to_datetime(start_date)) &
             (df["Date"] <= pd.to_datetime(end_date))]

    # ==================== CÁLCULO DE KPIs ====================
    total_bookings = len(dff)
    completed = len(dff[~dff["Is_Cancelled"]])
    cancelled = len(dff[dff["Is_Cancelled"]])
    cancel_rate = round((cancelled / total_bookings * 100), 1) if total_bookings > 0 else 0
    avg_value = round(dff["Booking Value"].mean(), 0) if total_bookings > 0 else 0
    avg_distance = round(dff["Ride Distance"].mean(), 1) if total_bookings > 0 else 0
    avg_rating = round(dff["Driver Ratings"].mean(), 1) if total_bookings > 0 else 0

    # ==================== TARJETAS KPI MEJORADAS ====================
    kpi_cards = dbc.Row([
        # Total de reservas
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-calendar-check fa-2x mb-2", 
                              style={'color': COLORS['primary']}),
                        html.H6("Total Reservas", className="card-title text-muted"),
                        html.H3(f"{total_bookings:,}", style={'color': COLORS['primary'], 'fontWeight': 'bold'})
                    ], className="text-center")
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=6, lg=2),
        
        # Reservas completadas
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-check-circle fa-2x mb-2", 
                              style={'color': COLORS['success']}),
                        html.H6("Completadas", className="card-title text-muted"),
                        html.H3(f"{completed:,}", style={'color': COLORS['success'], 'fontWeight': 'bold'})
                    ], className="text-center")
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=6, lg=2),
        
        # Tasa de cancelación
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-times-circle fa-2x mb-2", 
                              style={'color': COLORS['danger'] if cancel_rate > 15 else COLORS['warning']}),
                        html.H6("Cancelaciones", className="card-title text-muted"),
                        html.H3(f"{cancel_rate}%", 
                               style={'color': COLORS['danger'] if cancel_rate > 15 else COLORS['warning'], 
                                     'fontWeight': 'bold'})
                    ], className="text-center")
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=6, lg=2),
        
        # Precio promedio
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-dollar-sign fa-2x mb-2", 
                              style={'color': COLORS['primary']}),
                        html.H6("Precio Promedio", className="card-title text-muted"),
                        html.H3(f"${avg_value:,.0f}", style={'color': COLORS['primary'], 'fontWeight': 'bold'})
                    ], className="text-center")
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=6, lg=2),
        
        # Distancia promedio
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-route fa-2x mb-2", 
                              style={'color': COLORS['info']}),
                        html.H6("Distancia Promedio", className="card-title text-muted"),
                        html.H3(f"{avg_distance} km", style={'color': COLORS['info'], 'fontWeight': 'bold'})
                    ], className="text-center")
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=6, lg=2),
        
        # Calificación promedio
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-star fa-2x mb-2", 
                              style={'color': COLORS['warning']}),
                        html.H6("Calificación Promedio", className="card-title text-muted"),
                        html.H3(f"{avg_rating}/5", style={'color': COLORS['warning'], 'fontWeight': 'bold'})
                    ], className="text-center")
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=6, lg=2)
    ])

    # ==================== GRÁFICOS MEJORADOS ====================
    
    # 1. Estado de reservas (Pie chart con colores consistentes)
    status_colors = {
        'Completed': COLORS['success'],
        'Cancelled': COLORS['danger'], 
        'Incomplete': COLORS['warning']
    }
    
    fig_status = px.pie(dff, names="Booking Status", 
                       title="", 
                       color="Booking Status",
                       color_discrete_map=status_colors)
    
    # 2. Métodos de pago (Bar chart horizontal)
    payment_data = dff["Payment Method"].value_counts()
    fig_payment = px.bar(x=payment_data.values, y=payment_data.index, 
                        orientation='h',
                        title="",
                        color_discrete_sequence=[COLORS['primary']])
    fig_payment.update_layout(xaxis_title="Cantidad de Reservas", yaxis_title="")
    
    # 3. Distribución de calificaciones (Violin plot)
    fig_ratings = px.violin(dff, y="Driver Ratings", 
                           title="",
                           color_discrete_sequence=[COLORS['warning']])
    fig_ratings.update_layout(yaxis_title="Calificación (1-5)", xaxis_title="")
    
    # 4. Distribución de distancias (Histogram con KDE)
    fig_distance = px.histogram(dff, x="Ride Distance", 
                               title="",
                               marginal="box",  # Añadir boxplot marginal
                               color_discrete_sequence=[COLORS['info']])
    fig_distance.update_layout(xaxis_title="Distancia (km)", yaxis_title="Frecuencia")
    
    # 5. Tendencia temporal con línea de regresión
    trend_data = dff.groupby("Date").size().reset_index(name="Bookings")
    
    fig_trend = go.Figure()
    
    # Datos reales
    fig_trend.add_trace(go.Scatter(
        x=trend_data["Date"], 
        y=trend_data["Bookings"],
        mode='lines+markers',
        name='Reservas Diarias',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=4)
    ))
    
    # Línea de tendencia si está habilitada
    if "show_trend" in (trend_options or []):
        trend_values, r_squared = calculate_trend_line(trend_data["Date"], trend_data["Bookings"])
        fig_trend.add_trace(go.Scatter(
            x=trend_data["Date"], 
            y=trend_values,
            mode='lines',
            name=f'Tendencia (R² = {r_squared:.3f})',
            line=dict(color=COLORS['danger'], width=3, dash='dash')
        ))
    
    fig_trend.update_layout(
        xaxis_title="Fecha", 
        yaxis_title="Número de Reservas",
        title="",
        hovermode='x unified'
    )

    # ==================== CONFIGURACIÓN COMÚN DE GRÁFICOS ====================
    common_layout = dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['dark'], size=12),
        title_font=dict(size=16, color=COLORS['primary']),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    for fig in [fig_status, fig_payment, fig_ratings, fig_distance, fig_trend]:
        fig.update_layout(**common_layout)
    
    # Configuraciones específicas
    fig_status.update_traces(textposition='inside', textinfo='percent+label')
    fig_payment.update_traces(marker_color=COLORS['primary'])

    return kpi_cards, fig_status, fig_payment, fig_ratings, fig_distance, fig_trend

# ==================== EJECUTAR APLICACIÓN ====================
if __name__ == "__main__":
    print("🚀 Iniciando Dashboard de Análisis de Viajes...")
    print("📊 Dashboard optimizado para portafolio profesional")
    print("🔗 Accede en: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)