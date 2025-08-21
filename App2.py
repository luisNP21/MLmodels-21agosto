import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =======================
# Configuración general
# =======================
st.set_page_config(page_title="EDA CSV subido", page_icon="📊", layout="wide")
st.title("📊 EDA interactivo sobre CSV subido")
st.caption("Sube tu archivo CSV y explóralo con un EDA interactivo: tablas, estadísticas, correlación y gráficas (barras, histograma, dispersión y pastel).")

# =======================
# Sidebar: Carga del CSV
# =======================
st.sidebar.header("1) Cargar archivo CSV")
uploaded = st.sidebar.file_uploader("Arrastra o selecciona tu CSV", type=["csv"])

st.sidebar.subheader("Parámetros de lectura")
sep = st.sidebar.selectbox("Separador", options=[",", ";", "\\t"], index=0, help="\\t representa tabulador.")
decimal = st.sidebar.selectbox("Separador de decimales", options=[".", ","], index=0)
encoding = st.sidebar.selectbox("Encoding", options=["utf-8", "latin-1"], index=0)
parse_dates_opt = st.sidebar.checkbox("Intentar detectar fechas automáticamente", value=True)
dayfirst = st.sidebar.checkbox("Formato día/mes primero (dayfirst)", value=True)
na_values_text = st.sidebar.text_input("Valores NA (separados por comas)", value="NA,N/A,.,null,")

def detectar_y_parsear_fechas(df: pd.DataFrame, dayfirst: bool = True) -> pd.DataFrame:
    """Intenta convertir columnas tipo objeto a datetime cuando parezcan fechas."""
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df2[col], errors="raise", dayfirst=dayfirst)
                df2[col] = parsed
            except Exception:
                pass
    return df2

# =======================
# Lectura del CSV
# =======================
if uploaded is None:
    st.info("⬅️ Sube un archivo CSV desde la barra lateral para comenzar.")
    st.stop()

try:
    df_all = pd.read_csv(
        uploaded,
        sep="\t" if sep == "\\t" else sep,
        encoding=encoding,
        decimal=decimal,
        na_values=[s for s in na_values_text.split(",") if s != ""],
        engine="python",
    )
    if parse_dates_opt:
        df_all = detectar_y_parsear_fechas(df_all, dayfirst=dayfirst)
    st.success("✅ CSV leído correctamente.")
except Exception as e:
    st.error(f"❌ Error leyendo el CSV: {e}")
    st.stop()

if df_all.empty:
    st.warning("El archivo está vacío o no se pudo interpretar. Verifica el separador, decimales y encoding.")
    st.stop()

# =======================
# 2) Muestreo y selección de columnas
# =======================
st.sidebar.header("2) Muestreo y columnas")
max_muestra = int(min(500, len(df_all)))
tam_muestra = st.sidebar.slider("Tamaño de la muestra (máx. 500)", 10, max_muestra, max_muestra, step=10)
semilla_muestra = st.sidebar.number_input("Semilla de muestreo", min_value=0, value=123, step=1)

df_sampled = df_all.sample(n=tam_muestra, random_state=semilla_muestra) if len(df_all) > tam_muestra else df_all.copy()

todas_cols = df_sampled.columns.tolist()
default_cols = todas_cols[: min(6, len(todas_cols))]
cols_seleccion = st.sidebar.multiselect("Elige hasta 6 columnas", todas_cols, default=default_cols, max_selections=6)

if not cols_seleccion:
    st.warning("Selecciona al menos una columna en la barra lateral.")
    st.stop()

df = df_sampled[cols_seleccion].copy()

# =======================
# 3) Valores faltantes (opcional)
# =======================
st.sidebar.header("3) Valores faltantes (opcional)")
pct_na = st.sidebar.slider("Inyectar faltantes (%)", 0, 30, 0, step=5)
if pct_na > 0:
    mask = np.random.rand(*df.shape) < (pct_na / 100.0)
    df = df.mask(mask)

# Tipos de columnas
cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
cols_cat = df.select_dtypes(exclude=[np.number]).columns.tolist()

# =======================
# 4) Gráficas a mostrar
# =======================
st.sidebar.header("4) Gráficas")
graficas = st.sidebar.multiselect(
    "Selecciona los tipos de gráficos",
    ["Barras (categorías)", "Histograma (numérico)", "Dispersión (scatter)", "Pastel (categorías)", "Correlación (numérico)"],
    default=["Barras (categorías)", "Histograma (numérico)", "Dispersión (scatter)"]
)

# =======================
# Layout principal
# =======================
tab_datos, tab_resumen, tab_graficas = st.tabs(["📄 Datos", "📈 Resumen EDA", "🎨 Gráficas"])

with tab_datos:
    st.subheader("Vista de datos")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        label="⬇️ Descargar CSV (actual)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_actual.csv",
        mime="text/csv",
    )

with tab_resumen:
    st.subheader("Estadísticos básicos")
    if cols_num:
        st.markdown("**Numéricas**")
        st.dataframe(df[cols_num].describe().T, use_container_width=True)
    else:
        st.info("No hay columnas numéricas seleccionadas.")

    st.markdown("---")
    st.subheader("Distribuciones categóricas")
    if cols_cat:
        for c in cols_cat:
            st.markdown(f"**{c}**")
            vc = df[c].value_counts(dropna=False).rename_axis(c).reset_index(name="frecuencia")
            st.dataframe(vc, use_container_width=True)
    else:
        st.info("No hay columnas categóricas seleccionadas.")

    st.markdown("---")
    if "Correlación (numérico)" in graficas:
        st.subheader("Matriz de correlación (numéricas)")
        if len(cols_num) >= 2:
            corr = df[cols_num].corr(numeric_only=True)
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de correlación")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Se requieren al menos 2 columnas numéricas para la correlación.")

with tab_graficas:
    st.subheader("Constructor de gráficos")

    # Barras (categorías)
    if "Barras (categorías)" in graficas:
        st.markdown("### Barras")
        cat_col_bar = st.selectbox(
            "Variable categórica",
            options=["(ninguna)"] + cols_cat,
            index=0 if not cols_cat else 1,
            key="bar_cat"
        )
        if cat_col_bar and cat_col_bar != "(ninguna)":
            vc = df[cat_col_bar].value_counts(dropna=False).reset_index()
            vc.columns = [cat_col_bar, "frecuencia"]
            fig_bar = px.bar(vc, x=cat_col_bar, y="frecuencia", title=f"Conteo por {cat_col_bar}")
            st.plotly_chart(fig_bar, use_container_width=True)
        elif not cols_cat:
            st.info("No hay columnas categóricas para mostrar en barras.")

    # Histograma (numérico)
    if "Histograma (numérico)" in graficas:
        st.markdown("### Histograma")
        num_col_hist = st.selectbox(
            "Variable numérica",
            options=["(ninguna)"] + cols_num,
            index=0 if not cols_num else 1,
            key="hist_num"
        )
        bins = st.slider("Número de bins", 5, 60, 25, key="hist_bins")
        if num_col_hist and num_col_hist != "(ninguna)":
            fig_hist = px.histogram(df, x=num_col_hist, nbins=bins, title=f"Histograma de {num_col_hist}")
            st.plotly_chart(fig_hist, use_container_width=True)
        elif not cols_num:
            st.info("No hay columnas numéricas para mostrar en histograma.")

    # Dispersión (scatter)
    if "Dispersión (scatter)" in graficas:
        st.markdown("### Dispersión")
        if len(cols_num) >= 2:
            x_scatter = st.selectbox("Eje X (numérico)", options=cols_num, key="scat_x")
            y_scatter = st.selectbox("Eje Y (numérico)", options=[c for c in cols_num if c != x_scatter], key="scat_y")
            color_scatter = st.selectbox(
                "Color (categórico opcional)",
                options=["(ninguno)"] + cols_cat,
                key="scat_color"
            )
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
        cat_col_pie = st.selectbox(
            "Variable categórica",
            options=["(ninguna)"] + cols_cat,
            index=0 if not cols_cat else 1,
            key="pie_cat"
        )
        if cat_col_pie and cat_col_pie != "(ninguna)":
            vc = df[cat_col_pie].value_counts(dropna=False).reset_index()
            vc.columns = [cat_col_pie, "frecuencia"]
            fig_pie = px.pie(vc, names=cat_col_pie, values="frecuencia", title=f"Distribución de {cat_col_pie}")
            st.plotly_chart(fig_pie, use_container_width=True)
        elif not cols_cat:
            st.info("No hay columnas categóricas para mostrar en pastel.")

with st.expander("ℹ️ Consejos de uso"):
    st.markdown("""
- Sube tu **CSV** desde la barra lateral y ajusta **Separador**, **Decimales**, **Encoding** y **detección de fechas** si ves datos raros.
- Limita la **muestra** a 500 filas para un EDA rápido y selecciona **hasta 6 columnas**.
- Activa las **gráficas** que quieras y configúralas en la pestaña **Gráficas**.
- Descarga el CSV transformado desde la pestaña **Datos**.
""")
