"""
Observatorio de Exoneraciones del Servicio Social | INEN
Dashboard Streamlit sobre las prefacturas con descuento autorizado por el Servicio Social (2016-2026):
resumen, tendencias, nivel de exoneración, pacientes, territorio, asistentes sociales, control y alertas,
proyecciones, tabla dinámica y metodología. Misma arquitectura visual que el Observatorio del Consumo por IAFAS.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data_processing import COND_LABEL, GEO_ORDER, TRAMO_ORDER, apply_filters, load_fact, month_index
import panels as P
import asistente as A
from ui import CSS, STRETCH, Opts, mes, num

st.set_page_config(page_title="Observatorio de Exoneraciones del Servicio Social | INEN", page_icon="🤝", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

fact = load_fact()
MIN_M, MAX_M = fact["tiempo"].min(), fact["tiempo"].max()
LAST_DAY = fact["fecha"].max()

st.markdown(
    f"""<div class="app-header"><h1>Observatorio de Exoneraciones del Servicio Social</h1>
<p>Instituto Nacional de Enfermedades Neoplásicas · Prefacturas con descuento social · serie mensual {mes(MIN_M)} a {mes(MAX_M)}
(datos al {LAST_DAY:%d/%m/%Y})</p></div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Panel de control")
    asist_box = st.container()        # asistente de consultas: se llena cuando ya existen los filtros
    st.divider()
    st.markdown("**Filtros**")
    st.caption("Los filtros se aplican a todas las pestañas.")
    _meses = list(pd.date_range(MIN_M, MAX_M, freq="MS"))
    rng = st.select_slider("Rango de meses", options=_meses, value=(_meses[0], _meses[-1]), format_func=mes)
    excl_last = st.checkbox("Excluir el último mes de la serie", value=LAST_DAY.day < 28,
                            help=f"{mes(MAX_M)} llega solo hasta el {LAST_DAY:%d/%m/%Y}.")
    tipo_sel = st.multiselect("Tipo de historia (vacío = todos)", sorted(fact["tipo_hc"].unique().tolist()), default=[])
    cond_all = fact["condicion"].value_counts().index.tolist()
    cond_sel = st.multiselect("Condición del Servicio Social (vacío = todas)", cond_all, default=[], format_func=lambda x: COND_LABEL.get(x, x))
    tramo_sel = st.multiselect("Tramo de descuento (vacío = todos)", TRAMO_ORDER, default=[])
    with st.expander("Filtros territoriales y de personal"):
        geo_sel = st.multiselect("Ámbito territorial", [g for g in GEO_ORDER if g in set(fact["ambito_geo"].astype(str))], default=[])
        dep_sel = st.multiselect("Departamento de residencia", sorted(fact["departamento"].unique().tolist()), default=[])
        anon = st.checkbox("Mostrar asistentes con código (AS-01, AS-02...)", value=False,
                           help="Reemplaza los nombres por un código estable, ordenado por volumen histórico. Útil para presentar fuera de la unidad.")
        if anon:
            order = fact["asistente"].value_counts().index.tolist()
            amap = {a: ("Sin asistente registrado" if a == "Sin asistente registrado" else f"AS-{i + 1:02d}") for i, a in enumerate(o for o in order if o != "Sin asistente registrado")}
            amap["Sin asistente registrado"] = "Sin asistente registrado"
            fact = fact.assign(asistente=fact["asistente"].astype(str).map(amap).astype("category"))
        as_sel = st.multiselect("Asistente social", sorted(fact["asistente"].unique().tolist()), default=[])

    st.divider()
    st.markdown("**Visualización**")
    chart_type = st.radio("Tipo de gráfico", ["Línea", "Barras"], horizontal=True)
    markers = st.checkbox("Mostrar marcadores", value=True)
    labels = st.checkbox("Mostrar etiquetas de datos", value=False)
    lowess = st.checkbox("Línea suavizada (LOWESS)", value=False)
    log_scale = st.checkbox("Escala logarítmica", value=False)
    alpha = st.select_slider("Umbral de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05, format_func=lambda x: str(x).replace(".", ","))

date_lo, date_hi = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
if excl_last and date_hi >= MAX_M:
    date_hi = MAX_M - pd.DateOffset(months=1)
opts = Opts(date_lo=date_lo, date_hi=date_hi, chart_type=chart_type, markers=markers, labels=labels, lowess=lowess, log=log_scale, alpha=alpha)

flt = apply_filters(fact, date_lo=date_lo, date_hi=date_hi, tipo_hc=tipo_sel or None, condicion=cond_sel or None, tramo=tramo_sel or None,
                    ambito_geo=geo_sel or None, deptos=dep_sel or None, asistentes=as_sel or None)
n_filters = sum(bool(x) for x in (tipo_sel, cond_sel, tramo_sel, geo_sel, dep_sel, as_sel))
st.caption(f"**Selección:** {mes(date_lo)} a {mes(date_hi)} · "
           f"{'sin filtros adicionales' if n_filters == 0 else f'{n_filters} filtro(s) adicional(es) activo(s)'} · {len(flt):,} prefacturas")
with st.sidebar, asist_box:
    A.render_input(flt, fact)
A.render_resultado()               # resultado de la consulta en la sección principal, con línea de separación debajo

if flt.empty:
    st.warning("La selección no devuelve datos. Amplía el rango o quita filtros.")
    st.stop()
idx = month_index(flt["tiempo"].min(), date_hi)

NAV = ["📌 Resumen", "📈 Tendencias", "🎚️ Nivel de exoneración", "🧑‍🤝‍🧑 Pacientes", "🗺️ Territorio", "👩‍💼 Asistentes sociales",
       "🚩 Control y alertas", "🔮 Proyecciones", "🧮 Tabla dinámica", "📖 Metodología"]
choice = st.segmented_control("Pestañas", NAV, default=NAV[0], key="nav", label_visibility="collapsed") or NAV[0]
st.divider()

if choice == NAV[0]:
    st.subheader("Resumen del descuento social")
    P.panel_resumen(flt, idx, opts)
elif choice == NAV[1]:
    st.subheader("Tendencias")
    P.panel_tendencias(flt, idx, opts)
elif choice == NAV[2]:
    st.subheader("Nivel de exoneración")
    P.panel_nivel(flt, idx, opts)
elif choice == NAV[3]:
    st.subheader("Pacientes beneficiados")
    P.panel_pacientes(flt, fact, idx, opts)
elif choice == NAV[4]:
    st.subheader("Distribución territorial")
    P.panel_territorio(flt, opts)
elif choice == NAV[5]:
    st.subheader("Asistentes sociales")
    P.panel_asistentes(flt, idx, opts)
elif choice == NAV[6]:
    st.subheader("Control y alertas para auditoría")
    P.panel_alertas(flt, opts)
elif choice == NAV[7]:
    st.subheader("Proyecciones")
    P.panel_proyecciones(flt, idx, opts)
elif choice == NAV[8]:
    st.subheader("Tabla dinámica y descarga")
    dims = {"Año": "anio", "Mes": "tiempo", "Tipo de historia": "tipo_hc", "Condición": "condicion", "Tramo de descuento": "tramo",
            "% de descuento (escala)": "porc_escala", "Asistente social": "asistente", "Ámbito territorial": "ambito_geo",
            "Departamento": "departamento", "Provincia": "provincia", "Distrito": "distrito", "Documento": "doc_estado", "Responsable de cuenta": "responsable"}
    vals = {"Monto exonerado (S/)": "exonerado", "Monto total (S/)": "monto_total", "Monto pagado (S/)": "pagado", "Prefacturas": "n", "Pacientes únicos": "pid"}
    c = st.columns([2, 2, 1.2, 1.2])
    rows = c[0].multiselect("Filas", list(dims), default=["Año", "Tramo de descuento"])
    cols = c[1].multiselect("Columnas (opcional)", [x for x in dims if x not in rows], default=[])
    val = c[2].selectbox("Valor", list(vals))
    topk = c[3].number_input("Máx. filas", 10, 5000, 500, 50)
    if not rows:
        st.info("Elige al menos una dimensión en filas.")
    else:
        keys = [dims[x] for x in rows + cols]
        gb = flt.groupby(keys, observed=True)
        g = gb.size() if vals[val] == "n" else gb["pid"].nunique() if vals[val] == "pid" else gb[vals[val]].sum()
        g = g.rename("valor")
        if cols:
            tb = g.unstack([dims[x] for x in cols]).fillna(0)
            if vals[val] != "pid":
                tb["Total"] = tb.sum(axis=1)
        else:
            tb = g.to_frame(val)
        tb = tb.sort_values(tb.columns[-1], ascending=False).head(int(topk)).reset_index()
        if "tiempo" in tb.columns:
            tb["tiempo"] = pd.to_datetime(tb["tiempo"]).dt.strftime("%Y-%m")
        tb.columns = [" · ".join(map(str, c_)) if isinstance(c_, tuple) else str(c_) for c_ in tb.columns]
        inv = {v: k for k, v in dims.items()}
        tb = tb.rename(columns=inv)
        _dec = 0 if vals[val] in ("n", "pid") else 1
        _fm = {c_: (lambda v: num(v, _dec)) for c_ in tb.columns if pd.api.types.is_numeric_dtype(tb[c_]) and c_ != "Año"}
        st.dataframe(tb.style.format(_fm, na_rep="—"), hide_index=True, **STRETCH, height=520)
        st.download_button("Descargar CSV", tb.to_csv(index=False).encode("utf-8-sig"), file_name="tabla_dinamica_servicio_social.csv", mime="text/csv")
        if vals[val] == "pid":
            st.caption("Los pacientes únicos no se suman entre filas: un mismo paciente puede aparecer en varias.")
else:
    st.subheader("Metodología y notas")
    st.markdown(f"""
**Fuente.** Consulta sobre `INEN.HOS_PRE_FACTURA_CAB_TMP` (prefacturas activas, `IND_ESTADO = '1'`, con `VAL_MONTO_DSCTO > 0`) entre el 01/01/2016 y el
{LAST_DAY:%d/%m/%Y}. Cada fila es una prefactura que recibió descuento del Servicio Social. El asistente social sale de `SSO_REGISTRO_EXONERACIONES_MOV`
y el ubigeo y la dirección de la ficha socioeconómica (`SSO_FICHA_EVA_SOCIOECONO_MOV`). El archivo se compacta con `build_data.py`.

**Privacidad.** No se publican nombres, DNI, historia clínica ni dirección. El paciente se reemplaza por un seudónimo (hash con sal del tipo y número de historia) que
permite contar pacientes y seguir su recurrencia sin identificarlos. Se conserva el número de prefactura para rastrear casos en el sistema. Los nombres de los asistentes
se pueden reemplazar por un código (barra lateral).

**Montos.** Monto total = `VAL_MONTO_TOTAL` (valor de la prefactura a tarifa INEN). Exonerado = `VAL_MONTO_DSCTO` (descuento otorgado; es ingreso que el INEN deja de percibir,
un subsidio institucional). Pagado = total − exonerado. Se verificó que las tres cifras cuadran en todas las filas. El % exonerado ponderado es la suma del exonerado
entre la suma del monto total; difiere del promedio simple de `T_DSCTO_PORC` porque los descuentos altos se aplican sobre todo a prefacturas pequeñas.

**Fecha.** Es `AUD_FEC_ULTIMA_MODIFICACION` truncada al día: la fecha de la última modificación de la prefactura, no necesariamente la de atención o la de autorización
del descuento. Una prefactura modificada después de su emisión se ubica en el mes de la modificación. {mes(MAX_M)} es parcial y por defecto se excluye.

**Atención.** Paciente distinto en un mismo día. La consulta original suma los descuentos del paciente por día (`T_DSCTO`), que se usa como referencia pero no se repite
en los cálculos para evitar doble conteo.

**Condición.** Código `COD_CONDICION_SS` decodificado en la consulta (01 HOS, 02 ASE, 03 CL-P, 04 CL-H, 05 SIS, 06 DES, 07 SOC, 08 EXO). Casi todas las prefacturas son
HOS o ASE; el resto tiene menos de 20 casos.

**Escala y tramos.** Se considera escala habitual a los porcentajes enteros que aparecen en al menos 0,5% de las prefacturas (20, 25, 30, 35, 40, 50, 60 y 90%); el resto
se marca «fuera de escala». Tramos: hasta 20%, 21-30%, 31-40%, 41-59%, 60-89% y 90% a más.

**Territorio.** Ubigeo de la ficha socioeconómica vigente del paciente (residencia declarada en la última evaluación, no al momento de cada prefactura). Lima Metropolitana
= provincia de Lima; «Lima (sin provincia)» = ficha que solo registra el departamento. «Extranjero» = residencia declarada en América, Europa o Asia. El mapa usa los
límites departamentales de `data/peru_departamentos.geojson`.

**Asistentes sociales.** El gráfico de embudo compara la proporción de prefacturas con descuento de 90% a más de cada asistente con la proporción institucional, con límites
binomiales de 95% (z = 1,96) y 99,8% (z = 3,09). Los asistentes atienden servicios distintos, así que las diferencias pueden reflejar el perfil de sus pacientes.

**Control y alertas.** Reglas de muestreo para auditoría (umbrales ajustables): exonerado alto por prefactura (P99 por defecto), descuento alto, muchas prefacturas del mismo
paciente en un día, porcentajes distintos al mismo paciente el mismo día, exonerado anual alto por paciente y porcentaje fuera de escala; y cuatro reglas de calidad de registro
(sin asistente, sin DNI, registro en fin de semana, sin ubigeo). Una alerta no implica irregularidad.

**Tendencias.** Mann-Kendall con corrección de autocorrelación (Hamed-Rao, respaldo Yue-Wang) y pendiente de Sen sobre la serie mensual; APC por regresión log-lineal;
regresión lineal sobre años completos; quiebre automático con test de Chow (p-valor optimista porque la fecha se busca en los datos). La pandemia (mar–dic 2020) puede excluirse.

**Proyecciones.** Holt con tendencia amortiguada y banda por bootstrap de residuos (300 réplicas), un escenario conservador (media de tres meses) y uno lineal con la pendiente
de Sen. El cierre del año suma lo observado y los meses proyectados restantes.

**Limitaciones.** Solo incluye prefacturas con descuento: no permite estimar qué proporción de pacientes evaluados por el Servicio Social recibió descuento ni el total facturado
del INEN. No trae diagnóstico, servicio ni tipo de prestación de la prefactura. El 28% de las prefacturas no tiene ubigeo.

**Actualizar los datos.** `python build_data.py RUTA/DATA_SERVICIO_SOCIAL.xlsx` regenera `data/fact_ss.parquet`.
""")

# ---------------------------------------------------------------------------
# Autoría y cómo citar
# ---------------------------------------------------------------------------
_YR = 2026
_URL = "https://[completar-url].streamlit.app/"
_T_ES = "Observatorio de Exoneraciones del Servicio Social del INEN"
_T_EN = "INEN Social Service Fee Waiver Observatory"
st.divider()
st.markdown("""<p class="source-note" style="text-align:center; margin-top:0.6rem;">© Luis A. Orrego Ferreyros, Econ.,
MCE, MMD, PhD(c), CQRM · Epidemiólogo y Economista de la Salud — INEN</p>""", unsafe_allow_html=True)
st.markdown(
    f"""<div class="citation-box">
<h4>REFERENCIA WEB (formato AMA)</h4>
<p>Orrego-Ferreyros LA. {_T_ES}. Instituto Nacional de Enfermedades Neoplásicas. Published {_YR}. Accessed [Mes Día, Año]. {_URL}</p>
<h4>WEB REFERENCE (AMA format)</h4>
<p>Orrego-Ferreyros LA. {_T_EN}. National Institute of Neoplastic Diseases. Published {_YR}. Accessed [Month Day, Year]. {_URL}</p>
</div>""",
    unsafe_allow_html=True,
)
_ris = (f"TY  - ELEC\nAU  - Orrego-Ferreyros, Luis Alexander\nTI  - {_T_ES} / {_T_EN}\nPY  - {_YR}\n"
        f"PB  - Instituto Nacional de Enfermedades Neoplásicas\nUR  - {_URL}\nER  - \n")
_bib = (f"@misc{{orregoferreyros{_YR}servsocial,\n  author = {{Orrego-Ferreyros, Luis Alexander}},\n  title = {{{_T_ES} / {_T_EN}}},\n"
        f"  year = {{{_YR}}},\n  publisher = {{Instituto Nacional de Enfermedades Neopl\\'asicas}},\n  url = {{{_URL}}}\n}}\n")
_d1, _d2 = st.columns(2)
_d1.download_button("⬇️ BibTeX (*.bib)", _bib, file_name="observatorio_servicio_social_inen.bib", mime="text/plain", key="cite_bib")
_d2.download_button("⬇️ RIS (*.ris)", _ris, file_name="observatorio_servicio_social_inen.ris", mime="application/x-research-info-systems", key="cite_ris")
