"""panels.py: paneles del Observatorio de Exoneraciones del Servicio Social (INEN)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_processing import (
    ESCALA_ORDER, GEO_ORDER, METRICS, MONEY, TRAMO_ORDER, agg_table, gini, load_geo, patient_table, series_metric,
)
from stats_analysis import (
    apc_loglinear, cagr, detect_breakpoint_auto, linear_trend, mann_kendall_trend, project_series, significance_label,
    smooth_series,
)
from ui import (
    ACCENT, BAD, OK, PALETTE, PRIMARY, SECONDARY, STRETCH, WARN, MESES_ES, Opts, fmt_df, hex_to_rgba, interp, mes, num,
    pct, ptxt, pval, shades_of, soles, soles_short, style_fig, time_figure, warn,
)

TRAMO_COLOR = dict(zip(TRAMO_ORDER, ["#9DB9D3", "#5C8DB8", PRIMARY, "#C9A13B", SECONDARY, BAD]))
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def csv_button(df: pd.DataFrame, label: str, fname: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=fname, mime="text/csv", key=key)


def complete_years(idx: pd.DatetimeIndex) -> list[int]:
    s = pd.Series(1, index=idx)
    c = s.groupby(idx.year).size()
    return [int(y) for y, n in c.items() if n == 12]


def fmt_metric(v: float, metric: str) -> str:
    if metric in MONEY:
        return soles_short(v)
    if metric == "pct_exo":
        return pct(v)
    return num(v, 0)


def short_pid(p) -> str:
    return "P-" + format(int(p), "x")[-7:].upper()


def _cond_shift(d: pd.DataFrame, y0: int, y1: int) -> str:
    """Frase sobre el cambio de peso de la condición principal entre dos años."""
    ct = pd.crosstab(d["anio"], d["condicion"].astype(str), normalize="index") * 100
    if y0 not in ct.index or y1 not in ct.index or ct.shape[1] < 2:
        return ""
    dl = (ct.loc[y1] - ct.loc[y0]).sort_values()
    c = dl.index[-1]
    if dl.iloc[-1] < 5:
        return ""
    return (f"Parte del cambio es de composición: la condición <b>{c}</b> pasó de {num(ct.loc[y0, c], 1)}% a {num(ct.loc[y1, c], 1)}% de las prefacturas "
            "y tiene una escala de descuento propia (ver el mapa de calor de abajo).")


def drop_covid(s: pd.Series, flag: bool) -> pd.Series:
    if not flag:
        return s
    return s[~((s.index >= "2020-03-01") & (s.index <= "2020-12-01"))]


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
def panel_resumen(d: pd.DataFrame, idx: pd.DatetimeIndex, o: Opts) -> None:
    tot, exo, pag = d["monto_total"].sum(), d["exonerado"].sum(), d["pagado"].sum()
    npac = d["pid"].nunique()
    nat = len(d.drop_duplicates(["pid", "fecha"]))
    k = st.columns(4)
    k[0].metric("Monto exonerado", soles_short(exo), help="Suma de VAL_MONTO_DSCTO: ingreso que el INEN deja de percibir por el descuento social.")
    k[1].metric("Monto total prefacturado", soles_short(tot), help="Valor a tarifa INEN de las prefacturas que recibieron descuento.")
    k[2].metric("% exonerado (ponderado)", pct(exo / tot * 100 if tot else np.nan), help="Exonerado / monto total. Pondera por el tamaño de cada prefactura.")
    k[3].metric("Pagado por los pacientes", soles_short(pag))
    k = st.columns(5)
    k[0].metric("Prefacturas con descuento", num(len(d)))
    k[1].metric("Pacientes beneficiados", num(npac))
    k[2].metric("Atenciones (paciente-día)", num(nat))
    k[3].metric("Exonerado por paciente", soles(exo / npac if npac else np.nan))
    k[4].metric("Asistentes sociales", num(d.loc[d["asistente"] != "Sin asistente registrado", "asistente"].nunique()))

    st.markdown("##### Valor mensual de las prefacturas: parte exonerada y parte pagada")
    wide = pd.DataFrame({"Exonerado": series_metric(d, "exonerado", "MS", idx), "Pagado por el paciente": series_metric(d, "pagado", "MS", idx)})
    fig = time_figure(wide, o, stacked=True, colors={"Exonerado": SECONDARY, "Pagado por el paciente": hex_to_rgba(PRIMARY, .55)}, height=380,
                      allow_lowess=False, force_kind="bar")
    st.plotly_chart(fig, **STRETCH)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Pacientes beneficiados por mes y % exonerado")
        pm = series_metric(d, "pacientes", "MS", idx)
        pe = series_metric(d, "pct_exo", "MS", idx)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=pm.index, y=pm.values, name="Pacientes", marker_color=hex_to_rgba(PRIMARY, .7)), secondary_y=False)
        fig.add_trace(go.Scatter(x=pe.index, y=pe.values, name="% exonerado", line=dict(color=SECONDARY, width=2)), secondary_y=True)
        style_fig(fig, 340)
        fig.update_yaxes(title_text="Pacientes", secondary_y=False)
        fig.update_yaxes(title_text="% exonerado", ticksuffix="%", tickformat=".0f", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, **STRETCH)
    with c2:
        st.markdown("##### Composición del exonerado por tipo de historia y condición")
        dc = d.assign(condicion=np.where(d["condicion"].isin(["HOS", "ASE"]), d["condicion"].astype(str), "Otras"))
        a = agg_table(dc, ["tipo_hc", "condicion"])
        a = a[a["exonerado"] > 0]
        fig = go.Figure()
        conds = a.groupby("condicion", observed=True)["exonerado"].sum().sort_values(ascending=False).index.tolist()
        for i, cnd in enumerate(conds):
            x = a[a["condicion"] == cnd]
            fig.add_trace(go.Bar(x=x["tipo_hc"].astype(str), y=x["exonerado"], name=f"Condición {cnd}",
                                 marker_color={"HOS": SECONDARY, "ASE": PRIMARY}.get(str(cnd), "#9AA5B1")))
        fig.update_layout(barmode="stack", hovermode="closest")
        style_fig(fig, 340)
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
        st.plotly_chart(fig, **STRETCH)

    st.markdown("#### Resumen anual")
    an = agg_table(d, "anio").sort_values("anio")
    meses = d.groupby("anio")["tiempo"].nunique()
    an["meses"] = an["anio"].map(meses)
    an["var_exo"] = an["exonerado"].pct_change() * 100
    an["exo_mensual"] = an["exonerado"] / an["meses"]
    an["Año"] = an["anio"].astype(str) + np.where(an["meses"] < 12, " (parcial)", "")
    show = an[["Año", "meses", "prefacturas", "pacientes", "atenciones", "monto_total", "exonerado", "pagado", "pct_exo", "exo_paciente", "exo_mensual", "var_exo"]]
    show.columns = ["Año", "Meses", "Prefacturas", "Pacientes", "Atenciones", "Monto total", "Exonerado", "Pagado", "% exonerado",
                    "Exonerado por paciente", "Exonerado mensual promedio", "Var. % exonerado"]
    st.dataframe(fmt_df(show, money=["Monto total", "Exonerado", "Pagado", "Exonerado por paciente", "Exonerado mensual promedio"],
                        ints=["Meses", "Prefacturas", "Pacientes", "Atenciones"], pcts=["% exonerado", "Var. % exonerado"]), hide_index=True, **STRETCH)
    csv_button(show, "Descargar resumen anual (CSV)", "ss_resumen_anual.csv", "dl_res_an")

    full = an[an["meses"] == 12]
    if len(full) >= 2:
        f0, f1 = full.iloc[0], full.iloc[-1]
        g = cagr(f0["exonerado"], f1["exonerado"], f1["anio"] - f0["anio"])
        txt = (f"Entre {int(f0['anio'])} y {int(f1['anio'])} (años completos) el monto exonerado pasó de <b>{soles_short(f0['exonerado'])}</b> a "
               f"<b>{soles_short(f1['exonerado'])}</b>, un crecimiento compuesto de <b>{pct(g * 100) if g is not None else 'n/d'}</b> al año. "
               f"Los pacientes beneficiados pasaron de {num(f0['pacientes'])} a {num(f1['pacientes'])} y el exonerado por paciente de "
               f"{soles(f0['exo_paciente'])} a {soles(f1['exo_paciente'])}, de modo que el aumento se explica ")
        rp = f1["pacientes"] / f0["pacientes"]
        rs = f1["exo_paciente"] / f0["exo_paciente"]
        txt += ("sobre todo por <b>más pacientes</b>." if abs(np.log(rp)) > abs(np.log(rs)) else "sobre todo por <b>mayor exoneración por paciente</b>.")
        txt += (f" En todo el periodo seleccionado, el descuento cubrió el {pct(exo / tot * 100)} del valor de las prefacturas: por cada sol prefacturado "
                f"el paciente pagó {num(pag / tot, 2)} y el INEN asumió {num(exo / tot, 2)}.")
        st.markdown(interp(txt), unsafe_allow_html=True)
    if (an["meses"] < 12).any():
        st.caption("Los años marcados como parciales no tienen los doce meses en la selección; compara el exonerado mensual promedio, no el total.")


# ---------------------------------------------------------------------------
# Tendencias
# ---------------------------------------------------------------------------
def panel_tendencias(d: pd.DataFrame, idx: pd.DatetimeIndex, o: Opts) -> None:
    c = st.columns([1.4, 1])
    metr_lbl = c[0].selectbox("Métrica a evaluar", list(METRICS), key="t_m")
    excl_covid = c[1].checkbox("Excluir mar–dic 2020 (pandemia)", value=False, key="t_covid")
    metric = METRICS[metr_lbl]
    s = drop_covid(series_metric(d, metric, "MS", idx), excl_covid).dropna()
    if len(s) < 8:
        st.info(f"Se necesitan al menos 8 meses para evaluar tendencia; hay {len(s)}.")
        return
    mk = mann_kendall_trend(s.values, o.alpha)
    ya = series_metric(d, metric, "YS", idx)
    yrs = [y for y in complete_years(idx) if not (excl_covid and y == 2020)]
    ya = ya[ya.index.year.isin(yrs)]
    cg = cagr(float(ya.iloc[0]), float(ya.iloc[-1]), int(ya.index.year[-1] - ya.index.year[0])) if len(ya) >= 2 and ya.iloc[-1] > 0 else None
    ap = apc_loglinear(s.values)
    k = st.columns(5)
    if mk:
        icon = {"creciente": "📈", "decreciente": "📉"}.get(mk.trend, "➖")
        k[0].metric("Mann-Kendall", f"{icon} {mk.trend}")
        k[1].metric("p-valor", pval(mk.p_value), significance_label(mk.p_value, o.alpha), delta_color="off")
        k[2].metric("Pendiente de Sen", (soles_short(mk.sen_slope) if metric in MONEY else f"{mk.sen_slope:,.2f}") + " /mes")
    k[3].metric("CAGR (años completos)", pct(cg * 100) if cg is not None else "n/d",
                help="(último año completo / primer año completo)^(1/años) − 1. Solo usa los extremos.")
    k[4].metric("APC (log-lineal)", pct(ap.apc) if ap else "n/d",
                help="Cambio porcentual anual con ln(y) = a + b·t, como en Joinpoint." + (f" IC 95%: {pct(ap.lo)} a {pct(ap.hi)}; {ptxt(ap.p_value)}." if ap else ""))
    if mk:
        txt = (f"La serie mensual de <b>{metr_lbl.lower()}</b> muestra tendencia <b>{mk.trend}</b> (Mann-Kendall {mk.variant_used}, {ptxt(mk.p_value)}; "
               f"{significance_label(mk.p_value, o.alpha).lower()} con α = {o.alpha}). La pendiente de Sen es el cambio mensual típico, poco sensible a meses extremos. ")
        if ap:
            txt += f"El APC estimado es {pct(ap.apc)} al año (IC 95%: {pct(ap.lo)} a {pct(ap.hi)}), ajustado sobre todos los meses y no solo sobre los años extremos."
        st.markdown(interp(txt), unsafe_allow_html=True)

    x = np.arange(len(s))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=metr_lbl, mode="lines+markers" if o.markers else "lines", line=dict(color=PRIMARY, width=2)))
    if mk:
        fig.add_trace(go.Scatter(x=s.index, y=np.median(s.values - mk.sen_slope * x) + mk.sen_slope * x, name="Tendencia de Sen",
                                 line=dict(color=SECONDARY, width=3, dash="dash")))
    sm = smooth_series(np.nan_to_num(s.values.astype(float)))
    if sm is not None and o.lowess:
        fig.add_trace(go.Scatter(x=s.index, y=sm, name="LOWESS", line=dict(color=OK, width=2.5)))
    style_fig(fig, 400, f"{metr_lbl}: serie mensual y tendencia")
    if metric in MONEY:
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
    if metric == "pct_exo":
        fig.update_yaxes(ticksuffix="%")
    if o.log:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, **STRETCH)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("##### Tendencia anual (años completos)")
        if len(ya) >= 3:
            lt = linear_trend(ya.index.year.values, ya.values)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ya.index.year.astype(str), y=ya.values, marker_color=hex_to_rgba(PRIMARY, .7), name="Observado"))
            fig.add_trace(go.Scatter(x=ya.index.year.astype(str), y=lt.y_pred, name="Regresión lineal", line=dict(color=SECONDARY, width=3, dash="dash")))
            style_fig(fig, 320)
            fig.update_layout(hovermode="closest")
            if metric in MONEY:
                fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
            st.plotly_chart(fig, **STRETCH)
            st.caption(f"Pendiente {fmt_metric(lt.slope, metric)} por año · R² = {lt.r2:.2f} · {ptxt(lt.p_value)}. Con {len(ya)} puntos es una guía, no una prueba.")
        else:
            st.info("Se necesitan al menos 3 años completos.")
    with a2:
        st.markdown("##### Estacionalidad (índice por mes del año)")
        if len(s) >= 24:
            ratio = s / s.rolling(12, center=True, min_periods=6).mean().replace(0, np.nan)
            seas = ratio.groupby(ratio.index.month).mean()
            fig = go.Figure(go.Bar(x=[MESES_ES[m - 1] for m in seas.index], y=seas.values,
                                   marker_color=[SECONDARY if v > 1 else PRIMARY for v in seas.values]))
            fig.add_hline(y=1, line_dash="dot", line_color="#888")
            style_fig(fig, 320)
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, **STRETCH)
            st.markdown(interp(f"El mes típicamente más alto es <b>{MESES_ES[int(seas.idxmax()) - 1]}</b> ({(seas.max() - 1) * 100:+.0f}%) y el más bajo "
                               f"<b>{MESES_ES[int(seas.idxmin()) - 1]}</b> ({(seas.min() - 1) * 100:+.0f}%) respecto del promedio móvil anual."), unsafe_allow_html=True)
        else:
            st.info("Se necesitan al menos 24 meses.")

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("##### Cambio de nivel o de pendiente")
        auto = detect_breakpoint_auto(s, min_segment=6)
        if auto is None:
            st.info("Serie demasiado corta para buscar un quiebre.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="markers", marker=dict(color=PRIMARY, size=5), name="Observado"))
            fig.add_trace(go.Scatter(x=auto.x_before, y=auto.y_pred_before, name="Antes", line=dict(color=SECONDARY, width=2)))
            fig.add_trace(go.Scatter(x=auto.x_after, y=auto.y_pred_after, name="Después", line=dict(color=OK, width=2)))
            style_fig(fig, 320, f"Quiebre automático: {mes(auto.date)}")
            st.plotly_chart(fig, **STRETCH)
            st.caption(f"Pendiente antes {fmt_metric(auto.slope_before, metric)}/mes, después {fmt_metric(auto.slope_after, metric)}/mes · Chow F = {auto.f_stat:.2f}, "
                       f"{ptxt(auto.p_value)}. La fecha se busca en los datos, así que el p-valor es optimista.")
    with b2:
        st.markdown("##### Distribución por día de la semana (fecha de registro)")
        dw = d.groupby("dia_semana").agg(prefacturas=("num_prefactura", "size"), exonerado=("exonerado", "sum")).reindex(range(7), fill_value=0)
        fig = go.Figure(go.Bar(x=DIAS, y=dw["prefacturas"], marker_color=[PRIMARY] * 5 + [WARN, BAD], text=dw["prefacturas"], texttemplate="%{text:,.0f}"))
        style_fig(fig, 320)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, **STRETCH)
        st.caption(f"Sábado y domingo concentran el {pct(dw.loc[[5, 6], 'prefacturas'].sum() / max(dw['prefacturas'].sum(), 1) * 100)} de las prefacturas. "
                   "La fecha es la de última modificación de la prefactura, no necesariamente la de atención.")

    st.markdown("##### Variación interanual (mismo periodo del año)")
    last = d["tiempo"].max()
    ytd = d[d["tiempo"].dt.month <= last.month]
    t = agg_table(ytd, "anio").sort_values("anio")[["anio", "prefacturas", "pacientes", "exonerado", "pct_exo", "exo_paciente"]]
    for c_ in ["prefacturas", "pacientes", "exonerado", "exo_paciente"]:
        t[f"var_{c_}"] = t[c_].astype(float).pct_change() * 100
    t["anio"] = t["anio"].astype(str)
    t = t.rename(columns={"anio": "Año", "prefacturas": "Prefacturas", "pacientes": "Pacientes", "exonerado": "Exonerado", "pct_exo": "% exonerado",
                          "exo_paciente": "Exonerado por paciente", "var_prefacturas": "Var. % prefacturas", "var_pacientes": "Var. % pacientes",
                          "var_exonerado": "Var. % exonerado", "var_exo_paciente": "Var. % exonerado/paciente"})
    st.caption(f"Cada año se corta en Ene–{MESES_ES[last.month - 1]} para comparar periodos equivalentes con el último año disponible.")
    st.dataframe(fmt_df(t, money=["Exonerado", "Exonerado por paciente"], ints=["Prefacturas", "Pacientes"],
                        pcts=["% exonerado", "Var. % prefacturas", "Var. % pacientes", "Var. % exonerado", "Var. % exonerado/paciente"]),
                 hide_index=True, **STRETCH)


# ---------------------------------------------------------------------------
# Nivel de exoneración
# ---------------------------------------------------------------------------
def panel_nivel(d: pd.DataFrame, idx: pd.DatetimeIndex, o: Opts) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Porcentaje de descuento otorgado (escala)")
        e = agg_table(d, "porc_escala").set_index("porc_escala").reindex(ESCALA_ORDER).dropna(subset=["prefacturas"]).reset_index()
        e["pct_pref"] = e["prefacturas"] / e["prefacturas"].sum() * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=e["porc_escala"].astype(str), y=e["pct_pref"], name="% de prefacturas", marker_color=PRIMARY))
        fig.add_trace(go.Bar(x=e["porc_escala"].astype(str), y=e["pct_del_exonerado"], name="% del monto exonerado", marker_color=SECONDARY))
        style_fig(fig, 340)
        fig.update_layout(barmode="group", hovermode="x unified")
        fig.update_xaxes(type="category")
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, **STRETCH)
    with c2:
        st.markdown("##### Mezcla de tramos de descuento por año (% de prefacturas)")
        m = d.groupby(["anio", "tramo"], observed=True).size().unstack(fill_value=0).reindex(columns=TRAMO_ORDER, fill_value=0)
        m = m.div(m.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for t in TRAMO_ORDER:
            fig.add_trace(go.Bar(x=m.index.astype(str), y=m[t], name=t, marker_color=TRAMO_COLOR[t]))
        fig.update_layout(barmode="stack")
        style_fig(fig, 340)
        fig.update_yaxes(ticksuffix="%", range=[0, 100])
        st.plotly_chart(fig, **STRETCH)

    st.markdown("##### % exonerado ponderado, promedio simple y prefacturas con 90% a más, por mes")
    pe = series_metric(d, "pct_exo", "MS", idx)
    mn = d.groupby("tiempo")["porc"].mean().reindex(idx)
    p90 = ((d["porc"] >= 89.99).groupby(d["tiempo"]).mean() * 100).reindex(idx)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=p90.index, y=p90.values, name="% de prefacturas con 90% a más (eje derecho)", marker_color=hex_to_rgba(BAD, .22)), secondary_y=True)
    fig.add_trace(go.Scatter(x=pe.index, y=pe.values, name="% exonerado ponderado por monto", line=dict(color=SECONDARY, width=2.5)))
    fig.add_trace(go.Scatter(x=mn.index, y=mn.values, name="Promedio simple del % por prefactura", line=dict(color=PRIMARY, width=2, dash="dot")))
    style_fig(fig, 360)
    fig.update_yaxes(ticksuffix="%", tickformat=".0f", secondary_y=False)
    fig.update_yaxes(ticksuffix="%", tickformat=".0f", secondary_y=True, showgrid=False, rangemode="tozero")
    st.plotly_chart(fig, **STRETCH)
    mix = d.groupby(["anio", "tramo"], observed=True).size().unstack(fill_value=0).reindex(columns=TRAMO_ORDER, fill_value=0)
    mix = mix.div(mix.sum(axis=1), axis=0) * 100
    if len(mix) >= 2:
        y0, y1 = mix.index.min(), mix.index.max()
        dlt = (mix.loc[y1] - mix.loc[y0]).sort_values()
        st.markdown(interp(
            f"Entre {y0} y {y1} el tramo que más peso ganó fue <b>{dlt.index[-1]}</b> ({num(mix.loc[y0, dlt.index[-1]], 1)}% a {num(mix.loc[y1, dlt.index[-1]], 1)}% de las prefacturas) "
            f"y el que más perdió fue <b>{dlt.index[0]}</b> ({num(mix.loc[y0, dlt.index[0]], 1)}% a {num(mix.loc[y1, dlt.index[0]], 1)}%). Un cambio brusco de la mezcla entre dos años "
            "suele responder a una modificación de la escala o del procedimiento de calificación socioeconómica más que a un cambio en el perfil de los pacientes; conviene contrastarlo "
            "con la normativa interna vigente en cada periodo. " + _cond_shift(d, y0, y1)), unsafe_allow_html=True)
    pw, pm_ = d["exonerado"].sum() / d["monto_total"].sum() * 100, d["porc"].mean()
    st.markdown(interp(
        f"El promedio simple del descuento por prefactura es {pct(pm_)}, pero el % ponderado por monto es {pct(pw)}. "
        + ("Que el ponderado sea menor indica que los descuentos más altos se aplican a prefacturas de menor valor, y las prefacturas grandes reciben descuentos moderados."
           if pw < pm_ else "Que el ponderado sea mayor indica que los descuentos altos se concentran en prefacturas de mayor valor.")),
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Tramo de descuento según condición y tipo de historia (% de prefacturas)")
        h = d.assign(grupo=d["condicion"].astype(str) + " · " + d["tipo_hc"].astype(str))
        top = h["grupo"].value_counts()
        h = h[h["grupo"].isin(top[top >= 30].index)]
        hm = h.groupby(["grupo", "tramo"], observed=True).size().unstack(fill_value=0).reindex(columns=TRAMO_ORDER, fill_value=0)
        hm = hm.div(hm.sum(axis=1), axis=0) * 100
        fig = go.Figure(go.Heatmap(z=hm.values, x=TRAMO_ORDER, y=hm.index, colorscale="Blues", text=np.round(hm.values, 1),
                                   texttemplate="%{text}%", hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>", showscale=False))
        style_fig(fig, 340)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, **STRETCH)
        st.caption("Solo grupos con al menos 30 prefacturas. La condición es el código del Servicio Social (COD_CONDICION_SS).")
    with c2:
        st.markdown("##### Valor de la prefactura según tramo de descuento")
        fig = go.Figure()
        for t in TRAMO_ORDER:
            x = d.loc[d["tramo"] == t, "monto_total"]
            if len(x):
                fig.add_trace(go.Box(y=x, name=t, marker_color=TRAMO_COLOR[t], boxpoints=False))
        style_fig(fig, 340)
        fig.update_layout(showlegend=False, hovermode="closest")
        fig.update_yaxes(type="log", tickprefix="S/ ", title_text="Monto total (escala log)")
        st.plotly_chart(fig, **STRETCH)
        st.caption("Escala logarítmica: la caja va del percentil 25 al 75 y la línea interior es la mediana.")

    st.markdown("##### Resumen por tramo")
    t = agg_table(d, "tramo").set_index("tramo").reindex(TRAMO_ORDER).dropna(subset=["prefacturas"]).reset_index()
    t = t[["tramo", "prefacturas", "pacientes", "monto_total", "exonerado", "pct_exo", "exo_prefactura", "pct_del_exonerado"]]
    t.columns = ["Tramo", "Prefacturas", "Pacientes", "Monto total", "Exonerado", "% exonerado", "Exonerado por prefactura", "% del exonerado"]
    st.dataframe(fmt_df(t, money=["Monto total", "Exonerado", "Exonerado por prefactura"], ints=["Prefacturas", "Pacientes"], pcts=["% exonerado", "% del exonerado"]),
                 hide_index=True, **STRETCH)


# ---------------------------------------------------------------------------
# Pacientes
# ---------------------------------------------------------------------------
def panel_pacientes(d: pd.DataFrame, fact: pd.DataFrame, idx: pd.DatetimeIndex, o: Opts) -> None:
    pt = patient_table(d)
    v = pt["exonerado"].values
    n = len(pt)
    srt = np.sort(v)[::-1]
    share = lambda q: srt[: max(1, int(np.ceil(n * q)))].sum() / srt.sum() * 100
    g = gini(v)
    k = st.columns(5)
    k[0].metric("Pacientes", num(n))
    k[1].metric("Coeficiente de Gini", num(g, 2), help="0 = todos reciben lo mismo; 1 = un solo paciente recibe todo.")
    k[2].metric("% del exonerado en el top 1%", pct(share(.01)), help="Parte del monto exonerado que reciben el 1% de pacientes con mayor beneficio.")
    k[3].metric("% del exonerado en el top 10%", pct(share(.10)))
    k[4].metric("Mediana por paciente", soles(np.median(v)), help=f"Percentil 90: {soles(np.percentile(v, 90))}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Curva de concentración (Lorenz)")
        asc = np.sort(v)
        xs = np.linspace(0, 100, len(asc) + 1)
        ys = np.concatenate([[0], np.cumsum(asc) / asc.sum() * 100])
        step = max(1, len(xs) // 600)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs[::step], y=ys[::step], name="Exonerado acumulado", fill="tozeroy", line=dict(color=PRIMARY, width=2),
                                 fillcolor=hex_to_rgba(PRIMARY, .15)))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Igualdad", line=dict(color="#999", dash="dot")))
        style_fig(fig, 340)
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title_text="% de pacientes (de menor a mayor exonerado)", ticksuffix="%")
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, **STRETCH)
    with c2:
        st.markdown("##### Distribución del exonerado acumulado por paciente")
        fig = go.Figure(go.Histogram(x=np.log10(np.clip(v, 1, None)), nbinsx=50, marker_color=hex_to_rgba(PRIMARY, .75)))
        tv = [1, 2, 3, 4, 5]
        fig.update_xaxes(tickvals=tv, ticktext=["S/ 10", "S/ 100", "S/ 1 mil", "S/ 10 mil", "S/ 100 mil"], title_text="Exonerado por paciente (escala log)")
        style_fig(fig, 340)
        fig.update_layout(hovermode="closest", showlegend=False)
        st.plotly_chart(fig, **STRETCH)
    st.markdown(interp(
        f"El 10% de pacientes con mayor beneficio concentra el <b>{pct(share(.10))}</b> del monto exonerado y el 1% concentra el {pct(share(.01))}. "
        f"Un Gini de {num(g, 2)} indica una concentración {'alta' if g > .6 else 'moderada' if g > .4 else 'baja'}: el descuento sigue a pacientes con tratamientos caros "
        "o prolongados, por lo que el control del gasto pasa más por revisar los casos grandes que por el volumen de casos pequeños."), unsafe_allow_html=True)

    st.markdown("##### Pacientes nuevos y recurrentes por año")
    first = fact.groupby("pid")["anio"].min()
    py = d.drop_duplicates(["pid", "anio"])[["pid", "anio"]]
    py["tipo"] = np.where(py["pid"].map(first) == py["anio"], "Nuevo (primer año con descuento)", "Recurrente (con descuento en años previos)")
    nt = py.groupby(["anio", "tipo"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for i, col in enumerate(nt.columns):
        fig.add_trace(go.Bar(x=nt.index.astype(str), y=nt[col], name=col, marker_color=[SECONDARY, PRIMARY][i % 2]))
    fig.update_layout(barmode="stack")
    style_fig(fig, 340)
    st.plotly_chart(fig, **STRETCH)
    st.caption(f"Nuevo = primer año del paciente en toda la serie ({int(fact['anio'].min())} en adelante); los pacientes de {int(fact['anio'].min())} "
               "aparecen todos como nuevos porque no hay datos anteriores.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Años con descuento por paciente")
        ya = pt["anios"].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=ya.index.astype(str), y=ya.values, marker_color=PRIMARY, text=ya.values, texttemplate="%{text:,.0f}",
                               textposition="outside", textangle=0, cliponaxis=False))
        style_fig(fig, 300)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, **STRETCH)
    with c2:
        st.markdown("##### Documento de identidad registrado")
        dd = d.drop_duplicates("pid")["doc_estado"].value_counts()
        fig = go.Figure(go.Pie(labels=dd.index.astype(str), values=dd.values, hole=.5, marker_colors=[PRIMARY, WARN, BAD, "#7D7D7D"]))
        style_fig(fig, 300)
        st.plotly_chart(fig, **STRETCH)

    st.markdown("##### Pacientes con mayor monto exonerado")
    topn = st.slider("Número de pacientes", 10, 200, 30, 10, key="pac_top")
    tb = pt.head(topn).copy()
    tb.insert(0, "Paciente", tb["pid"].map(short_pid))
    tb["% acumulado"] = tb["exonerado"].cumsum() / v.sum() * 100
    tb = tb[["Paciente", "exonerado", "monto_total", "pct_exo", "prefacturas", "dias", "anios", "porc_max", "primer", "ultimo", "% acumulado"]]
    tb.columns = ["Paciente", "Exonerado", "Monto total", "% exonerado", "Prefacturas", "Días con descuento", "Años", "% máximo", "Primera", "Última", "% acumulado"]
    for c_ in ("Primera", "Última"):
        tb[c_] = tb[c_].dt.strftime("%d/%m/%Y")
    st.dataframe(fmt_df(tb, money=["Exonerado", "Monto total"], ints=["Prefacturas", "Días con descuento", "Años"], pcts=["% exonerado", "% máximo", "% acumulado"]),
                 hide_index=True, **STRETCH)
    st.caption("El código de paciente es un seudónimo: no permite identificar a la persona. Para rastrear un caso, usa el número de prefactura en «Control y alertas».")


# ---------------------------------------------------------------------------
# Territorio
# ---------------------------------------------------------------------------
def panel_territorio(d: pd.DataFrame, o: Opts) -> None:
    sin = (d["ambito_geo"] == "Sin ubigeo").mean() * 100
    if sin > 5:
        st.markdown(warn(f"El {pct(sin)} de las prefacturas no tiene ubigeo en la ficha socioeconómica; el mapa y las tablas describen solo a quienes sí lo tienen."),
                    unsafe_allow_html=True)
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("##### Mapa por departamento de residencia")
        opt = {"Exonerado por paciente (S/)": "exo_paciente", "Monto exonerado (S/)": "exonerado", "Pacientes beneficiados": "pacientes", "% exonerado (ponderado)": "pct_exo"}
        ml = st.selectbox("Indicador", list(opt), key="geo_m")
        mc = opt[ml]
        dep = agg_table(d[~d["departamento"].isin(["Sin ubigeo", "Extranjero"])], "departamento")
        geo = load_geo()
        if geo is not None and len(dep):
            dep["key"] = dep["departamento"].astype(str).str.upper()
            fig = go.Figure(go.Choropleth(
                geojson=geo, featureidkey="properties.NOMBDEP", locations=dep["key"], z=dep[mc],
                colorscale="Blues", marker_line_color="white", marker_line_width=.6,
                text=dep["departamento"].astype(str), customdata=np.stack([dep["exonerado"], dep["pacientes"], dep["exo_paciente"]], axis=1),
                hovertemplate="<b>%{text}</b><br>Exonerado: S/ %{customdata[0]:,.0f}<br>Pacientes: %{customdata[1]:,.0f}<br>Por paciente: S/ %{customdata[2]:,.0f}<extra></extra>",
                colorbar=dict(thickness=12, len=.7),
            ))
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, **STRETCH)
            if mc in ("exonerado", "pacientes"):
                st.caption("Lima domina la escala en montos y pacientes; «Exonerado por paciente» compara la intensidad del beneficio entre regiones.")
    with c2:
        st.markdown("##### Por ámbito territorial")
        a = agg_table(d, "ambito_geo").set_index("ambito_geo").reindex(GEO_ORDER).dropna(subset=["prefacturas"]).reset_index()
        fig = go.Figure(go.Bar(y=a["ambito_geo"].astype(str), x=a["exonerado"], orientation="h", marker_color=PRIMARY,
                               text=[soles_short(x) for x in a["exonerado"]], textposition="outside", cliponaxis=False))
        style_fig(fig, 300)
        fig.update_xaxes(range=[0, a["exonerado"].max() * 1.25])
        fig.update_layout(hovermode="closest", yaxis=dict(autorange="reversed"))
        fig.update_xaxes(tickprefix="S/ ", tickformat=",.0f")
        st.plotly_chart(fig, **STRETCH)
        t = a[["ambito_geo", "pacientes", "exonerado", "exo_paciente", "pct_exo"]]
        t.columns = ["Ámbito", "Pacientes", "Exonerado", "Por paciente", "% exonerado"]
        st.dataframe(fmt_df(t, money=["Exonerado", "Por paciente"], ints=["Pacientes"], pcts=["% exonerado"]), hide_index=True, **STRETCH)
        st.caption("«Lima (sin provincia)»: la ficha registra solo el departamento. «Extranjero»: residencia declarada fuera del Perú.")

    st.markdown("##### Distritos de Lima Metropolitana y Callao")
    lm = d[d["ambito_geo"].isin(["Lima Metropolitana", "Callao"]) & (d["distrito"] != "Sin dato")]
    t = agg_table(lm, "distrito").head(30)
    if len(t):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=t["distrito"].astype(str), y=t["exonerado"], name="Exonerado", marker_color=PRIMARY), secondary_y=False)
        fig.add_trace(go.Scatter(x=t["distrito"].astype(str), y=t["exo_paciente"], name="Exonerado por paciente", mode="markers",
                                 marker=dict(color=SECONDARY, size=9)), secondary_y=True)
        style_fig(fig, 400)
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f", secondary_y=False)
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f", secondary_y=True, showgrid=False)
        fig.update_xaxes(tickangle=-45)
        fig.update_layout(margin=dict(b=150, l=60), legend=dict(y=1.08, yanchor="bottom"))
        st.plotly_chart(fig, **STRETCH)
        top3 = ", ".join(t["distrito"].astype(str).head(3))
        st.markdown(interp(f"Los distritos con mayor monto exonerado son <b>{top3}</b>, que reúnen el {pct(t['exonerado'].head(3).sum() / lm['exonerado'].sum() * 100)} "
                           "del exonerado de Lima Metropolitana y Callao. Coinciden con los distritos más poblados de los conos norte, este y sur."), unsafe_allow_html=True)

    st.markdown("##### Tabla por departamento")
    t = agg_table(d, "departamento")[["departamento", "prefacturas", "pacientes", "monto_total", "exonerado", "pct_exo", "exo_paciente", "pct_del_exonerado"]]
    t.columns = ["Departamento", "Prefacturas", "Pacientes", "Monto total", "Exonerado", "% exonerado", "Por paciente", "% del exonerado"]
    st.dataframe(fmt_df(t, money=["Monto total", "Exonerado", "Por paciente"], ints=["Prefacturas", "Pacientes"], pcts=["% exonerado", "% del exonerado"]),
                 hide_index=True, **STRETCH, height=420)
    csv_button(t, "Descargar (CSV)", "ss_departamentos.csv", "dl_dep")


# ---------------------------------------------------------------------------
# Asistentes sociales
# ---------------------------------------------------------------------------
def panel_asistentes(d: pd.DataFrame, idx: pd.DatetimeIndex, o: Opts) -> None:
    da = d[d["asistente"] != "Sin asistente registrado"]
    g = agg_table(da, "asistente")
    extra = da.groupby("asistente", observed=True).agg(
        p90=("porc", lambda x: (x >= 89.99).mean() * 100), fuera=("porc_escala", lambda x: (x == "Fuera de escala").mean() * 100),
        meses=("tiempo", "nunique"))
    g = g.merge(extra, left_on="asistente", right_index=True)
    g["pref_mes"] = g["prefacturas"] / g["meses"]
    min_n = st.slider("Mínimo de prefacturas para incluir a un asistente", 1, 500, 50, 10, key="as_min")
    g = g[g["prefacturas"] >= min_n]
    if g.empty:
        st.info("Ningún asistente alcanza el mínimo elegido.")
        return
    st.caption(f"{len(g)} asistentes con al menos {min_n} prefacturas. Prefacturas sin asistente registrado: {num((d['asistente'] == 'Sin asistente registrado').sum())}.")

    st.markdown("##### Prefacturas con 90% a más de descuento por asistente (gráfico de embudo)")
    ajuste = st.radio("Comparar contra", ["Lo esperado según su mezcla de casos (condición, tipo de historia y año)", "El promedio institucional sin ajuste"],
                      key="as_aj", help="Con ajuste, a cada prefactura se le asigna la proporción de 90% a más de su estrato (condición × tipo de historia × año); "
                                        "el esperado del asistente es la suma de esas proporciones (estandarización indirecta).")
    hi = (da["porc"] >= 89.99).astype(float)
    p0 = hi.mean()
    if ajuste.startswith("Lo esperado"):
        ps = hi.groupby([da["condicion"].astype(str), da["tipo_hc"].astype(str), da["anio"]]).transform("mean")
    else:
        ps = pd.Series(p0, index=da.index)
    ev = pd.DataFrame({"a": da["asistente"].astype(str), "o": hi, "e": ps, "v": ps * (1 - ps)}).groupby("a").sum()
    ev = ev.reindex(g["asistente"].astype(str))
    g = g.assign(O=ev["o"].values, E=ev["e"].values, V=ev["v"].values)
    g["razon"] = np.where(g["E"] > 0, g["O"] / g["E"], np.nan)
    zi = pd.Series(np.where(g["V"] > 0, (g["O"] - g["E"]) / np.sqrt(g["V"].clip(lower=1e-9)), 0.0), index=g.index)
    g["z"] = zi
    ee = np.linspace(max(1.0, g["E"].min() * .8), g["E"].max() * 1.05, 200)
    fig = go.Figure()
    for z, nm, col in ((1.96, "Límite 95%", WARN), (3.09, "Límite 99,8%", BAD)):
        se = np.sqrt((1 - p0) / ee)
        fig.add_trace(go.Scatter(x=ee, y=1 + z * se, name=nm, line=dict(color=col, dash="dot", width=1.5)))
        fig.add_trace(go.Scatter(x=ee, y=np.clip(1 - z * se, 0, None), showlegend=False, line=dict(color=col, dash="dot", width=1.5)))
    fig.add_hline(y=1, line_color="#888")
    colr = np.where(zi.abs() > 3.09, BAD, np.where(zi.abs() > 1.96, WARN, PRIMARY))
    fig.add_trace(go.Scatter(x=g["E"], y=g["razon"], mode="markers", name="Asistente", marker=dict(color=colr, size=10, line=dict(color="white", width=1)),
                             text=g["asistente"].astype(str), customdata=np.stack([g["O"], g["p90"], g["prefacturas"]], axis=1),
                             hovertemplate="<b>%{text}</b><br>Observadas: %{customdata[0]:,.0f} · esperadas: %{x:,.1f}<br>Razón O/E: %{y:.2f}"
                                           "<br>% con 90% a más: %{customdata[1]:.1f}% de %{customdata[2]:,.0f} prefacturas<extra></extra>"))
    style_fig(fig, 430)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Prefacturas con 90% a más esperadas")
    fig.update_yaxes(title_text="Razón observado / esperado")
    st.plotly_chart(fig, **STRETCH)
    n_hi, n_lo = int((zi > 3.09).sum()), int((zi < -3.09).sum())
    st.markdown(interp(
        "Cada punto es un asistente; 1 significa que otorga descuentos de 90% a más tantas veces como se espera. Las curvas marcan cuánto puede alejarse de 1 "
        f"solo por azar. Fuera del límite de 99,8%: <b>{n_hi}</b> por encima y <b>{n_lo}</b> por debajo de lo esperado. "
        + ("El ajuste por condición, tipo de historia y año descuenta que algunos asistentes atiendan grupos con descuentos más altos; lo que queda puede reflejar "
           "servicios o perfiles de pacientes que la base no registra. " if ajuste.startswith("Lo esperado") else
           "Sin ajuste, buena parte de la dispersión se debe a que cada asistente atiende grupos distintos (p. ej. condición HOS frente a ASE). ")
        + "Es una señal para muestrear casos, no una conclusión."), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Monto exonerado por asistente y año")
        top = g.head(25)["asistente"].astype(str).tolist()
        hm = da[da["asistente"].isin(top)].groupby(["asistente", "anio"], observed=True)["exonerado"].sum().unstack(fill_value=0).reindex(top)
        fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns.astype(str), y=hm.index.astype(str), colorscale="Blues",
                                   hovertemplate="%{y}<br>%{x}: S/ %{z:,.0f}<extra></extra>", colorbar=dict(thickness=10)))
        style_fig(fig, 620)
        fig.update_layout(hovermode="closest", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, **STRETCH)
    with c2:
        st.markdown("##### Distribución del % de descuento por asistente")
        top15 = g.sort_values("prefacturas", ascending=False).head(15)["asistente"].astype(str).tolist()
        fig = go.Figure()
        for a in top15:
            fig.add_trace(go.Box(x=da.loc[da["asistente"] == a, "porc"], name=a, marker_color=PRIMARY, boxpoints=False, orientation="h"))
        style_fig(fig, 620)
        fig.update_layout(showlegend=False, hovermode="closest", yaxis=dict(autorange="reversed"))
        fig.update_xaxes(ticksuffix="%", range=[0, 100])
        st.plotly_chart(fig, **STRETCH)
        st.caption("Los 15 asistentes con más prefacturas en la selección.")

    st.markdown("##### Tabla por asistente")
    t = g[["asistente", "prefacturas", "pacientes", "meses", "pref_mes", "monto_total", "exonerado", "pct_exo", "porc_mediano", "p90", "razon", "z", "fuera", "primer_mes", "ultimo_mes"]].copy()
    t.columns = ["Asistente", "Prefacturas", "Pacientes", "Meses activos", "Prefacturas por mes", "Monto total", "Exonerado", "% exonerado", "Mediana %",
                 "% con 90% a más", "Razón O/E (90% a más)", "z", "% fuera de escala", "Primer mes", "Último mes"]
    t["Razón O/E (90% a más)"] = t["Razón O/E (90% a más)"].map(lambda v: num(v, 2))
    st.dataframe(fmt_df(t, money=["Monto total", "Exonerado"], ints=["Prefacturas", "Pacientes", "Meses activos"], decs=["Prefacturas por mes", "z"],
                        pcts=["% exonerado", "Mediana %", "% con 90% a más", "% fuera de escala"], dates=["Primer mes", "Último mes"]), hide_index=True, **STRETCH, height=460)
    csv_button(t, "Descargar (CSV)", "ss_asistentes.csv", "dl_as")


# ---------------------------------------------------------------------------
# Control y alertas
# ---------------------------------------------------------------------------
def panel_alertas(d: pd.DataFrame, o: Opts) -> None:
    st.caption("Reglas de revisión sobre cada prefactura. Son criterios de muestreo para auditoría: una alerta no implica irregularidad.")
    c = st.columns(4)
    p99 = float(np.percentile(d["exonerado"], 99))
    u_exo = c[0].number_input("Exonerado alto (S/ por prefactura)", min_value=100.0, value=float(round(p99, -2)), step=100.0,
                              help=f"Por defecto el percentil 99 de la selección ({soles(p99)}).")
    u_pct = c[1].number_input("Descuento alto (%)", min_value=10, max_value=100, value=90, step=5)
    u_dia = c[2].number_input("Prefacturas del mismo paciente en un día", min_value=2, max_value=20, value=6, step=1)
    u_pac = c[3].number_input("Exonerado acumulado por paciente en el año (S/)", min_value=1000.0, value=20000.0, step=1000.0)
    pa = d.groupby(["pid", "anio"])["exonerado"].transform("sum")
    rules = {
        f"Exonerado ≥ {soles(u_exo)}": d["exonerado"] >= u_exo,
        f"Descuento ≥ {u_pct}%": d["porc"] >= u_pct - 0.01,
        f"Paciente con ≥ {u_dia} prefacturas el mismo día": d["n_pref_dia"] >= u_dia,
        "Distintos % de descuento al mismo paciente el mismo día": d["porc_distintos_dia"] >= 2,
        f"Paciente con exonerado anual ≥ {soles(u_pac)}": pa >= u_pac,
        "Porcentaje fuera de la escala habitual": d["porc_escala"] == "Fuera de escala",
        "Sin asistente social registrado": d["asistente"] == "Sin asistente registrado",
        "Paciente sin DNI (otro documento o ninguno)": d["doc_estado"] != "DNI",
        "Registrada en sábado o domingo": d["dia_semana"] >= 5,
        "Sin ubigeo en la ficha socioeconómica": d["ambito_geo"] == "Sin ubigeo",
    }
    rows = []
    for nm, m in rules.items():
        rows.append({"Regla": nm, "Prefacturas": int(m.sum()), "% prefacturas": m.mean() * 100, "Exonerado": d.loc[m, "exonerado"].sum(),
                     "% exonerado": d.loc[m, "exonerado"].sum() / d["exonerado"].sum() * 100, "Pacientes": d.loc[m, "pid"].nunique()})
    rt = pd.DataFrame(rows)
    st.dataframe(fmt_df(rt, money=["Exonerado"], ints=["Prefacturas", "Pacientes"], pcts=["% prefacturas", "% exonerado"]), hide_index=True, **STRETCH)

    score = sum(m.astype(int) for m in list(rules.values())[:6])
    d = d.assign(alertas=score)
    st.markdown("##### Prefacturas con varias alertas a la vez")
    sc = d["alertas"].value_counts().sort_index()
    cc = st.columns(len(sc))
    for i, (k_, v_) in enumerate(sc.items()):
        cc[i].metric(f"{k_} alerta(s)", num(v_))
    st.caption("El puntaje suma las seis primeras reglas (las de monto, porcentaje y repetición). Las cuatro últimas son de calidad de registro.")

    st.markdown("##### Detalle para revisión")
    c1, c2 = st.columns([2, 1])
    sel = c1.multiselect("Reglas a incluir (vacío = ordenar por puntaje)", list(rules), default=[], key="al_sel")
    modo = c2.radio("Combinar reglas", ["Cumple alguna", "Cumple todas"], horizontal=True, key="al_modo")
    if sel:
        ms = [rules[s] for s in sel]
        m = np.logical_or.reduce(ms) if modo == "Cumple alguna" else np.logical_and.reduce(ms)
        det = d[m]
    else:
        det = d[d["alertas"] >= 2]
    det = det.sort_values(["alertas", "exonerado"], ascending=False)
    st.caption(f"{num(len(det))} prefacturas · exonerado {soles(det['exonerado'].sum())}")
    show = det.head(2000).assign(Paciente=lambda x: x["pid"].map(short_pid))[
        ["num_prefactura", "fecha", "Paciente", "tipo_hc", "condicion", "asistente", "monto_total", "exonerado", "porc", "n_pref_dia", "alertas"]]
    show.columns = ["Prefactura", "Fecha", "Paciente", "Tipo de historia", "Condición", "Asistente", "Monto total", "Exonerado", "% descuento", "Prefacturas ese día", "Alertas"]
    show["Fecha"] = show["Fecha"].dt.strftime("%d/%m/%Y")
    st.dataframe(fmt_df(show, money=["Monto total", "Exonerado"], ints=["Prefacturas ese día", "Alertas"], pcts=["% descuento"]), hide_index=True, **STRETCH, height=460)
    csv_button(show, "Descargar detalle (CSV)", "ss_alertas_detalle.csv", "dl_al")
    if len(det) > 2000:
        st.caption("La tabla muestra las primeras 2 000 filas; acota con más reglas o con los filtros laterales.")

    if sel:
        st.markdown("##### Evolución mensual de la(s) regla(s) elegida(s)")
        ms = [rules[s] for s in sel]
        m = pd.Series(np.logical_or.reduce(ms) if modo == "Cumple alguna" else np.logical_and.reduce(ms), index=d.index)
        s = m.groupby(d["tiempo"]).mean() * 100
        fig = go.Figure(go.Scatter(x=s.index, y=s.values, line=dict(color=BAD, width=2), name="% de prefacturas con alerta"))
        style_fig(fig, 300)
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig, **STRETCH)


# ---------------------------------------------------------------------------
# Proyecciones
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calculando proyección...")
def _proj(vals: tuple, start: str, horizon: int, ci: float):
    s = pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="MS"))
    return project_series(s, horizon=horizon, n_boot=300, ci=ci)


def panel_proyecciones(d: pd.DataFrame, idx: pd.DatetimeIndex, o: Opts) -> None:
    c = st.columns([1.4, 1, 1, 1])
    opt = {k: v for k, v in METRICS.items() if v in ("exonerado", "monto_total", "prefacturas", "pacientes", "atenciones")}
    ml = c[0].selectbox("Métrica", list(opt), key="p_m")
    _to_dec = 12 - idx.max().month + 12
    hz = c[1].slider("Horizonte (meses)", 3, 24, int(min(max(_to_dec, 3), 24)), key="p_h", help="Por defecto, hasta diciembre del año siguiente.")
    ci = c[2].select_slider("Banda", options=[.8, .9, .95], value=.9, format_func=lambda x: f"{int(x * 100)}%", key="p_ci")
    desde = c[3].selectbox("Ajustar desde", [y for y in sorted(set(idx.year)) if y <= idx.year.max() - 2], index=0, key="p_desde")
    metric = opt[ml]
    s = series_metric(d, metric, "MS", idx)
    s = s[s.index.year >= desde]
    r = _proj(tuple(s.values.tolist()), str(s.index[0].date()), hz, ci)
    if r is None:
        st.info("Se necesitan al menos 12 meses para proyectar.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Observado", line=dict(color=PRIMARY, width=2)))
    fig.add_trace(go.Scatter(x=list(r.idx_future) + list(r.idx_future[::-1]), y=list(r.recommended_upper) + list(r.recommended_lower[::-1]),
                             fill="toself", fillcolor=hex_to_rgba(SECONDARY, .15), line=dict(width=0), name=f"Banda {int(ci * 100)}%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r.idx_future, y=r.recommended, name="Holt amortiguado (recomendado)", line=dict(color=SECONDARY, width=3)))
    fig.add_trace(go.Scatter(x=r.idx_future, y=r.conservative, name="Conservador (media 3 meses)", line=dict(color="#7D7D7D", dash="dot")))
    fig.add_trace(go.Scatter(x=r.idx_future, y=r.optimistic, name="Lineal (pendiente de Sen)", line=dict(color=OK, dash="dash")))
    style_fig(fig, 420, f"{ml}: proyección a {hz} meses")
    if metric in MONEY:
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
    st.plotly_chart(fig, **STRETCH)

    fut = pd.DataFrame({"Mes": r.idx_future, "Recomendado": r.recommended, "Inferior": r.recommended_lower, "Superior": r.recommended_upper,
                        "Conservador": r.conservative, "Lineal (Sen)": r.optimistic})
    yr = idx.max().year
    obs_y = s[s.index.year == yr].sum()
    fy = fut[fut["Mes"].dt.year == yr]
    k = st.columns(3)
    if len(fy):
        k[0].metric(f"Cierre estimado {yr}", fmt_metric(obs_y + fy["Recomendado"].sum(), metric),
                    help=f"Observado {fmt_metric(obs_y, metric)} + {len(fy)} mes(es) proyectado(s) con el escenario recomendado.")
    fy2 = fut[fut["Mes"].dt.year == yr + 1]
    if len(fy2) == 12:
        k[1].metric(f"Proyección {yr + 1}", fmt_metric(fy2["Recomendado"].sum(), metric),
                    help=f"Suma de las bandas mensuales: {fmt_metric(fy2['Inferior'].sum(), metric)} a {fmt_metric(fy2['Superior'].sum(), metric)} (aproximación conservadora).")
    last12 = s.iloc[-12:].sum()
    k[2].metric("Últimos 12 meses observados", fmt_metric(last12, metric))
    if metric in ("pacientes",):
        st.caption("Pacientes: la suma anual de pacientes mensuales cuenta a un mismo paciente en cada mes que recibe descuento (pacientes-mes).")
    st.markdown(interp(
        "El escenario recomendado (Holt con tendencia amortiguada) va reduciendo la pendiente para no sobrestimar a mediano plazo; la banda sale de remuestrear "
        "los residuos 300 veces. El conservador repite la media de los últimos tres meses y el lineal extrapola la pendiente de Sen. "
        "La proyección supone que la política de descuentos y la tarifa no cambian; un cambio en la escala o en el tarifario rompe el supuesto."), unsafe_allow_html=True)
    fs = fut.copy()
    fs["Mes"] = fs["Mes"].map(mes)
    st.dataframe(fmt_df(fs, money=["Recomendado", "Inferior", "Superior", "Conservador", "Lineal (Sen)"]) if metric in MONEY
                 else fmt_df(fs, ints=["Recomendado", "Inferior", "Superior", "Conservador", "Lineal (Sen)"]), hide_index=True, **STRETCH)
    csv_button(fut, "Descargar proyección (CSV)", "ss_proyeccion.csv", "dl_pr")
