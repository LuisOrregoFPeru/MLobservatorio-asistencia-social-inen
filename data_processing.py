"""
data_processing.py
-------------------
Carga, filtros y agregaciones del Observatorio de Exoneraciones del Servicio Social (INEN).

Fuente: data/fact_ss.parquet (generado con build_data.py). Una fila por prefactura con
descuento autorizado por el Servicio Social; el paciente está seudonimizado.

Definiciones
  - Monto total        = VAL_MONTO_TOTAL, valor de la prefactura a tarifa INEN.
  - Monto exonerado    = VAL_MONTO_DSCTO, descuento otorgado (ingreso que el INEN deja de percibir).
  - Monto pagado       = monto total − monto exonerado, lo que paga el paciente.
  - % exonerado        = exonerado / monto total. Ponderado: suma de exonerado / suma de monto total.
  - Atención           = paciente distinto en un día (varias prefacturas del mismo día cuentan una vez).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
FACT_FILE = DATA_DIR / "fact_ss.parquet"
GEO_FILE = DATA_DIR / "peru_departamentos.geojson"

TRAMO_ORDER = ["Hasta 20%", "21-30%", "31-40%", "41-59%", "60-89%", "90% a más"]
ESCALA_ORDER = ["20%", "25%", "30%", "35%", "40%", "50%", "60%", "90%", "Fuera de escala"]
GEO_ORDER = ["Lima Metropolitana", "Callao", "Lima provincias", "Lima (sin provincia)", "Otras regiones", "Extranjero", "Sin ubigeo"]
COND_LABEL = {"HOS": "HOS (cód. 01)", "ASE": "ASE (cód. 02)", "CL-P": "CL-P (cód. 03)", "CL-H": "CL-H (cód. 04)",
              "SIS": "SIS (cód. 05)", "DES": "DES (cód. 06)", "SOC": "SOC (cód. 07)", "EXO": "EXO (cód. 08)"}

METRICS = {
    "Monto exonerado (S/)": "exonerado",
    "Monto total prefacturado (S/)": "monto_total",
    "Monto pagado por el paciente (S/)": "pagado",
    "Prefacturas con descuento": "prefacturas",
    "Pacientes beneficiados": "pacientes",
    "Atenciones (paciente-día)": "atenciones",
    "% exonerado (ponderado)": "pct_exo",
    "Exonerado por paciente (S/)": "exo_paciente",
}
MONEY = {"exonerado", "monto_total", "pagado", "exo_paciente"}


@st.cache_data(show_spinner="Cargando prefacturas del Servicio Social...")
def load_fact() -> pd.DataFrame:
    df = pd.read_parquet(FACT_FILE)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["tiempo"] = pd.to_datetime(df["tiempo"])
    df["dia_semana"] = df["fecha"].dt.dayofweek.astype("int8")
    return df


@st.cache_data(show_spinner=False)
def load_geo() -> dict | None:
    import json
    if not GEO_FILE.exists():
        return None
    return json.loads(GEO_FILE.read_text(encoding="utf-8"))


def apply_filters(df: pd.DataFrame, *, date_lo=None, date_hi=None, tipo_hc=None, condicion=None, tramo=None,
                  ambito_geo=None, deptos=None, asistentes=None) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if date_lo is not None:
        m &= df["tiempo"] >= date_lo
    if date_hi is not None:
        m &= df["tiempo"] <= date_hi
    for col, val in (("tipo_hc", tipo_hc), ("condicion", condicion), ("tramo", tramo), ("ambito_geo", ambito_geo),
                     ("departamento", deptos), ("asistente", asistentes)):
        if val:
            m &= df[col].isin(val)
    return df[m]


def month_index(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(lo, hi, freq="MS")


def _period_key(d: pd.DataFrame, freq: str) -> pd.Series:
    return d["tiempo"] if freq == "MS" else pd.to_datetime(d["anio"].astype(str) + "-01-01")


def series_metric(d: pd.DataFrame, metric: str, freq: str = "MS", idx: pd.DatetimeIndex | None = None) -> pd.Series:
    """Serie mensual (MS) o anual (YS) de una métrica; los periodos sin datos valen cero."""
    if d.empty:
        return pd.Series(dtype=float)
    k = _period_key(d, freq)
    g = d.groupby(k, observed=True)
    if metric in ("exonerado", "monto_total", "pagado"):
        s = g[metric].sum()
    elif metric == "prefacturas":
        s = g.size().astype(float)
    elif metric == "pacientes":
        s = g["pid"].nunique().astype(float)
    elif metric == "atenciones":
        s = d.assign(_k=k).drop_duplicates(["_k", "pid", "fecha"]).groupby("_k").size().astype(float)
    elif metric == "pct_exo":
        s = g["exonerado"].sum() / g["monto_total"].sum().replace(0, np.nan) * 100
    elif metric == "exo_paciente":
        s = g["exonerado"].sum() / g["pid"].nunique().replace(0, np.nan)
    else:
        raise ValueError(metric)
    if idx is not None:
        full = idx if freq == "MS" else pd.DatetimeIndex(sorted({pd.Timestamp(f"{t.year}-01-01") for t in idx}))
        fill = np.nan if metric in ("pct_exo", "exo_paciente") else 0.0
        s = s.reindex(full, fill_value=fill)
    s.index.name = "tiempo"
    return s.astype(float)


def agg_table(d: pd.DataFrame, by) -> pd.DataFrame:
    """Tabla agregada por `by`: prefacturas, pacientes, atenciones, montos, % exonerado, exonerado por paciente, % y % acumulado."""
    if d.empty:
        return pd.DataFrame()
    by = [by] if isinstance(by, str) else list(by)
    g = d.groupby(by, observed=True).agg(
        prefacturas=("num_prefactura", "size"), pacientes=("pid", "nunique"),
        monto_total=("monto_total", "sum"), exonerado=("exonerado", "sum"), pagado=("pagado", "sum"),
        porc_mediano=("porc", "median"), primer_mes=("tiempo", "min"), ultimo_mes=("tiempo", "max"),
    )
    at = d.drop_duplicates(by + ["pid", "fecha"]).groupby(by, observed=True).size().rename("atenciones")
    g = g.join(at)
    g["pct_exo"] = g["exonerado"] / g["monto_total"].replace(0, np.nan) * 100
    g["exo_paciente"] = g["exonerado"] / g["pacientes"].replace(0, np.nan)
    g["exo_prefactura"] = g["exonerado"] / g["prefacturas"].replace(0, np.nan)
    g = g.sort_values("exonerado", ascending=False)
    tot = g["exonerado"].sum()
    g["pct_del_exonerado"] = g["exonerado"] / tot * 100 if tot else 0.0
    g["pct_acum"] = g["pct_del_exonerado"].cumsum()
    return g.reset_index()


def patient_table(d: pd.DataFrame) -> pd.DataFrame:
    """Una fila por paciente seudonimizado."""
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("pid").agg(
        exonerado=("exonerado", "sum"), monto_total=("monto_total", "sum"), prefacturas=("num_prefactura", "size"),
        dias=("fecha", "nunique"), anios=("anio", "nunique"), primer=("fecha", "min"), ultimo=("fecha", "max"),
        porc_max=("porc", "max"),
    )
    g["pct_exo"] = g["exonerado"] / g["monto_total"].replace(0, np.nan) * 100
    return g.sort_values("exonerado", ascending=False).reset_index()


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x >= 0]
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
