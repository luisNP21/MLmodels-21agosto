import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

# -------------------------
# Configuración general
# -------------------------
st.set_page_config(
    page_title="EDA Sintético de Deportes",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 EDA con Datos Sintéticos de Deportes")
st.caption("Genera un conjunto de datos deportivo simulado y realiza un análisis exploratorio interactivo.")

# -------------------------
# Sidebar: Controles
# -------------------------
st.sidebar.header("⚙️ Parámetros de generación")
seed = st.sidebar.number_input("Semilla (reproducibilidad)", min_value=0, value=42, step=1)
np.random.seed(seed)

n_filas = st.sidebar.slider("Número de filas (muestras)", min_value=50, max_value=500, value=200, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Selección de columnas (máx. 6)")
# Definimos un catálogo amplio de columnas para que el usuario escoja
columnas_catalogo = {
    # Cualitativas
    "deporte (cat)": "deporte",
    "equipo (cat)": "equipo",
    "posicion (cat)": "posicion",
    "pais (cat)": "pais",
    "sexo (cat)": "sexo",
    "lesionado (cat)": "lesionado",
    # Mixtas (fechas)
    "fecha_partido (fecha)": "fecha_partido",
    # Cuantitativas
    "edad (num)": "edad",
    "altura_cm (num)": "altura_cm",
    "peso_kg (num)": "peso_kg",
    "minutos_jugados (num)": "minutos_jugados",
    "puntos (num)": "puntos",
    "asistencias (num)": "asistencias",
    "rebotes (num)": "rebotes",
    "velocidad_max_kmh (num)": "velocidad_max_kmh",
    "salario_miles_usd (num)": "salario_miles_usd"
}

seleccion_columnas = st.sidebar.multiselect(
    "Elige hasta 6 columnas",
    list(columnas_catalogo.keys()),
    default=["deporte (cat)", "equipo (cat)", "edad (num)", "puntos (num)", "asistencias (num)", "rebotes (num)"],
    max_selections=6
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 Valores faltantes")
pct_na = st.sidebar.slider("Porcentaje de faltantes", 0, 30, 5, step=5, help="Porcentaje aproximado de celdas que se volverán NaN al azar.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Gráficas a mostrar")
graficas = st.sidebar.multiselect(
    "Selecciona los tipos de gráficos",
    ["Barras (categorías)", "Histograma (numérico)", "Dispersión (scatter)", "Pastel (categorías)", "Correlación (numérico)"],
    default=["Barras (categorías)", "Histograma (numérico)", "Dispersión (scatter)"]
)

# -------------------------
# Generación de datos sintéticos
# -------------------------
def generar_dataset(n):
    # Dominios deportivos y categóricos
    deportes = ["Fútbol", "Baloncesto", "Béisbol", "Tenis", "Atletismo", "Natación"]
    equipos = ["Tigres", "Leones", "Águilas", "Toros", "Pumas", "Lobos", "Halcones", "Tiburones"]
    posiciones = ["Portero", "Defensa", "Mediocampo", "Delantero", "Alero", "Base", "Lanzador", "Receptor"]
    paises = ["Colombia", "Argentina", "Brasil", "España", "EEUU", "México", "Francia", "Alemania"]
    sexo = ["M", "F"]

    # Señales numéricas (correlaciones suaves para que tenga sentido)
    edad = np.random.normal(25, 5, n).clip(16, 45).round(0)  # 16 - 45
    altura = np.random.normal(178, 10, n).clip(150, 210).round(1)  # cm
    peso = (altura - 100) + np.random.normal(10, 8, n)  # regla simple con ruido
    peso = np.clip(peso, 45, 120).round(1)

    minutos = np.random.normal(45, 20, n).clip(0, 120).round(0)
    # rendimiento base correlacionado con minutos y edad (en forma de campana)
    rendimiento = (minutos/3) + (-(edad-27)**2)/50 + np.random.normal(0, 5, n)
    puntos = np.clip(rendimiento + 15, 0, None).round(0)

    asistencias = (puntos * np.random.uniform(0.1, 0.4, n) + np.random.normal(0, 2, n)).clip(0, None).round(0)
    rebotes = (np.sqrt(puntos) * np.random.uniform(0.5, 2.0, n) + np.random.normal(0, 1, n)).clip(0, None).round(0)

    velocidad = np.random.normal(28, 4, n).clip(15, 40).round(1)  # km/h
    salario = (puntos * 3 + asistencias * 2 + rebotes * 1.5 + np.random.normal(0, 20, n)).clip(10, 500).round(1)  # miles USD

    # Categóricos
    df = pd.DataFrame({
        "deporte": np.random.choice(deportes, n),
        "equipo": np.random.choice(equipos, n),
        "posicion": np.random.choice(posiciones, n),
        "pais": np.random.choice(paises, n),
        "sexo": np.random.choice(sexo, n),
        "lesionado": np.where(np.random.rand(n) < 0.15, "Sí", "No"),
        "fecha_partido": pd.to_datetime(datetime(2023, 1, 1)) + pd.to_timedelta(np.random.randint(0, 900, n), unit="D"),
        "edad": edad.astype(int),
        "altura_cm": altura,
        "peso_kg": peso,
        "minutos_jugados": minutos.astype(int),
        "puntos": puntos.astype(int),
        "asistencias": asistencias.astype(int),
        "rebotes": rebotes.astype(int),
        "velocidad_max_kmh": velocidad,
        "salario_miles_usd": salario
    })
    return df

df_full = generar_dataset(n_filas)

# Restringir a las columnas elegidas (hasta 6)
if len(seleccion_columnas) == 0:
    st.warning("Selecciona al menos una columna en la barra lateral.")
    st.stop()

cols_finales = [columnas_catalogo[k] for k in seleccion_columnas]
df = df_full[cols_finales].copy()

# Inyectar NaN aleatorios según porcentaje elegido
if pct_na > 0:
    mask = np.random.rand(*df.shape) < (pct_na/100.0)
    df = df.mask(mask)

# Reconocer tipos
cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
cols_categoricas = df.select_dtypes(exclude=[np.number]).columns.tolist()

# -------------------------
# Layout principal: Pestañas
# -------------------------
tab_datos, tab_resumen, tab_graficas = st.tabs(["📄 Datos", "📈 Resumen EDA", "🎨 Gráficas"])

with tab_datos:
    st.subheader("Vista de datos")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        label="⬇️ Descargar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="deportes_sintetico.csv",
        mime="text/csv",
    )

with tab_resumen:
    st.subheader("Estadísticos básicos")
    if cols_numericas:
        st.markdown("**Numéricas**")
        st.dataframe(df[cols_numericas].describe().T, use_container_width=True)
    else:
        st.info("No hay columnas numéricas seleccionadas.")

    st.markdown("---")
    st.subheader("Distribuciones de categóricas")
    if cols_categoricas:
        for c in cols_categoricas:
            st.markdown(f"**{c}**")
            vc = df[c].value_counts(dropna=False).rename_axis(c).reset_index(name="frecuencia")
            st.dataframe(vc, use_container_width=True)
    else:
        st.info("No hay columnas categóricas seleccionadas.")

    st.markdown("---")
    st.subheader("Matriz de correlación (numéricas)")
    if len(cols_numericas) >= 2 and "Correlación (numérico)" in graficas:
        corr = df[cols_numericas].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Matriz de correlación"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    elif "Correlación (numérico)" in graficas:
        st.info("Se requieren al menos 2 columnas numéricas para la correlación.")

with tab_graficas:
    st.subheader("Constructor de gráficos")
    st.caption("Configura abajo según el tipo de gráfico seleccionado en la barra lateral.")

    # Barras (categorías)
    if "Barras (categorías)" in graficas:
        st.markdown("### Barras")
        cat_col_bar = st.selectbox("Variable categórica", options=cols_categoricas, index=0 if cols_categoricas else None, key="bar_cat")
        if cat_col_bar:
            vc = df[cat_col_bar].value_counts(dropna=False).reset_index()
            vc.columns = [cat_col_bar, "frecuencia"]
            fig_bar = px.bar(vc, x=cat_col_bar, y="frecuencia", title=f"Conteo por {cat_col_bar}")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Selecciona al menos una columna categórica para barras.")

    # Histograma (numérico)
    if "Histograma (numérico)" in graficas:
        st.markdown("### Histograma")
        num_col_hist = st.selectbox("Variable numérica", options=cols_numericas, index=0 if cols_numericas else None, key="hist_num")
        bins = st.slider("Número de bins", 5, 60, 25, key="hist_bins")
        if num_col_hist:
            fig_hist = px.histogram(df, x=num_col_hist, nbins=bins, title=f"Histograma de {num_col_hist}")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Selecciona al menos una columna numérica para histograma.")

    # Dispersión (scatter)
    if "Dispersión (scatter)" in graficas:
        st.markdown("### Dispersión")
        if len(cols_numericas) >= 2:
            x_scatter = st.selectbox("Eje X (numérico)", options=cols_numericas, key="scat_x")
            y_scatter = st.selectbox("Eje Y (numérico)", options=[c for c in cols_numericas if c != x_scatter], key="scat_y")
            color_scatter = st.selectbox("Color (opcional, categórico)", options=["(ninguno)"] + cols_categoricas, key="scat_color")
            kwargs = {}
            if color_scatter and color_scatter != "(ninguno)":
                kwargs["color"] = color_scatter
            fig_scatter = px.scatter(df, x=x_scatter, y=y_scatter, title=f"Dispersión {x_scatter} vs {y_scatter}", **kwargs)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Se requieren al menos 2 columnas numéricas para la dispersión.")

    # Pastel (categorías)
    if "Pastel (categorías)" in graficas:
        st.markdown("### Pastel")
        cat_col_pie = st.selectbox("Variable categórica", options=cols_categoricas, index=0 if cols_categoricas else None, key="pie_cat")
        if cat_col_pie:
            vc = df[cat_col_pie].value_counts(dropna=False).reset_index()
            vc.columns = [cat_col_pie, "frecuencia"]
            fig_pie = px.pie(vc, names=cat_col_pie, values="frecuencia", title=f"Distribución de {cat_col_pie}")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Selecciona al menos una columna categórica para pastel.")

# -------------------------
# Notas de uso
# -------------------------
with st.expander("ℹ️ Cómo usar"):
    st.markdown("""
1. **Elige la semilla** y el **número de filas** en la barra lateral.
2. Selecciona **hasta 6 columnas** (mezcla cuantitativas, cualitativas o fechas).
3. Ajusta el **porcentaje de faltantes** si quieres evaluar robustez del EDA.
4. Marca los **tipos de gráfica** a usar y configura cada una en la pestaña **Gráficas**.
5. Revisa **Estadísticos** y **Correlaciones** en la pestaña **Resumen EDA**.
6. **Descarga** el CSV desde la pestaña **Datos**.
""")
