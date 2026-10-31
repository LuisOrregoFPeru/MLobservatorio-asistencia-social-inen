"""
asistente.py
------------
Asistente «Pregunta a los datos» del Observatorio de Exoneraciones del Servicio Social (INEN).

La pregunta se escribe en la barra lateral (render_input) y el resultado se muestra en la sección
principal, en un recuadro sobre las pestañas (render_resultado).

Dos modos:
  - Modo IA (Claude): con ANTHROPIC_API_KEY en st.secrets, variable de entorno o ingresada en la barra
    lateral. El modelo no ejecuta código: llama a la herramienta `consultar_datos`, que solo hace
    agregaciones predefinidas con pandas. Al modelo le llegan tablas agregadas, nunca filas ni identificadores.
  - Modo básico (sin clave): intérprete por reglas en español (año, condición, tipo de historia, tramo,
    asistente, territorio, métrica, desglose y ranking).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

from ui import _V, PRIMARY, SECONDARY, STRETCH, mes, num

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_ROWS_LLM = 40
MAX_TOOL_ROUNDS = 6

DIMS = {
    "anio": "anio", "mes": "tiempo", "tipo_historia": "tipo_hc", "condicion": "condicion", "tramo": "tramo",
    "porcentaje": "porc_escala", "asistente": "asistente", "ambito_territorial": "ambito_geo",
    "departamento": "departamento", "provincia": "provincia", "distrito": "distrito", "documento": "doc_estado",
}
DIM_LABEL = {
    "anio": "Año", "mes": "Mes", "tipo_historia": "Tipo de historia", "condicion": "Condición", "tramo": "Tramo de descuento",
    "porcentaje": "% de descuento", "asistente": "Asistente social", "ambito_territorial": "Ámbito territorial",
    "departamento": "Departamento", "provincia": "Provincia", "distrito": "Distrito", "documento": "Documento",
}
METRICS = {
    "exonerado": "Monto exonerado (S/)", "monto_total": "Monto total (S/)", "pagado": "Monto pagado (S/)",
    "prefacturas": "Prefacturas", "pacientes": "Pacientes únicos", "atenciones": "Atenciones (paciente-día)",
    "exonerado_por_paciente": "Exonerado por paciente (S/)", "pct_exonerado": "% exonerado (ponderado)",
}
MONEY = {"exonerado", "monto_total", "pagado", "exonerado_por_paciente"}
ADITIVAS = {"exonerado", "monto_total", "pagado", "prefacturas", "atenciones"}

_FILTROS_LISTA = ["tipo_historia", "condicion", "tramo", "porcentaje", "asistente", "ambito_territorial", "departamento", "provincia", "distrito", "documento"]
TOOL = {
    "name": "consultar_datos",
    "description": (
        "Agrega las prefacturas con descuento del Servicio Social del INEN. Devuelve una tabla agrupada por las dimensiones pedidas, "
        "ordenada por la métrica, con el total general y el porcentaje de cada fila (solo métricas aditivas). Úsala tantas veces como haga falta. "
        "Los filtros de texto aceptan coincidencias parciales sin distinguir mayúsculas ni tildes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metrica": {"type": "string", "enum": list(METRICS), "description": "Por defecto exonerado."},
            "agrupar_por": {"type": "array", "items": {"type": "string", "enum": list(DIMS)}, "maxItems": 3},
            "top": {"type": "integer", "minimum": 1, "maximum": 150, "description": "Filas a devolver (por defecto 15)."},
            "orden": {"type": "string", "enum": ["desc", "asc", "natural"], "description": "natural = orden de la dimensión (series de tiempo)."},
            "filtros": {
                "type": "object",
                "properties": {
                    **{k: {"type": "array", "items": {"type": "string"}} for k in _FILTROS_LISTA},
                    "anio_desde": {"type": "integer"}, "anio_hasta": {"type": "integer"},
                    "mes_desde": {"type": "string", "description": "AAAA-MM"}, "mes_hasta": {"type": "string", "description": "AAAA-MM"},
                    "porc_min": {"type": "number", "description": "% de descuento mínimo por prefactura"},
                    "porc_max": {"type": "number", "description": "% de descuento máximo por prefactura"},
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Motor de consulta (solo agregaciones predefinidas)
# ---------------------------------------------------------------------------
def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()


def _match_values(col: pd.Series, wanted) -> list:
    cats = [str(c) for c in col.cat.categories] if hasattr(col, "cat") else col.astype(str).unique().tolist()
    out = []
    for w in wanted or []:
        nw = _norm(w)
        exact = [c for c in cats if _norm(c) == nw]
        out += exact or [c for c in cats if nw and nw in _norm(c)]
    return sorted(set(out))


def _agg(g: pd.DataFrame) -> dict:
    tot, exo = g["monto_total"].sum(), g["exonerado"].sum()
    pac = g["pid"].nunique()
    return {"exonerado": exo, "monto_total": tot, "pagado": g["pagado"].sum(), "prefacturas": len(g), "pacientes": pac,
            "atenciones": len(g.drop_duplicates(["pid", "fecha"])), "exonerado_por_paciente": exo / pac if pac else np.nan,
            "pct_exonerado": exo / tot * 100 if tot else np.nan}


def consultar(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    f = spec.get("filtros") or {}
    m = pd.Series(True, index=df.index)
    aplicados = []
    for key in _FILTROS_LISTA:
        if f.get(key):
            col = DIMS[key]
            vals = _match_values(df[col], f[key])
            m &= df[col].astype(str).isin(vals)
            aplicados.append(f"{DIM_LABEL[key]}: {', '.join(vals) if vals else '(sin coincidencias)'}")
    if f.get("anio_desde"):
        m &= df["anio"] >= int(f["anio_desde"]); aplicados.append(f"desde {f['anio_desde']}")
    if f.get("anio_hasta"):
        m &= df["anio"] <= int(f["anio_hasta"]); aplicados.append(f"hasta {f['anio_hasta']}")
    for key, op in (("mes_desde", ">="), ("mes_hasta", "<=")):
        if f.get(key):
            try:
                t = pd.Timestamp(str(f[key])[:7] + "-01")
                m &= (df["tiempo"] >= t) if op == ">=" else (df["tiempo"] <= t)
                aplicados.append(f"{'desde' if op == '>=' else 'hasta'} {mes(t)}")
            except Exception:  # noqa: BLE001
                pass
    if f.get("porc_min") is not None:
        m &= df["porc"] >= float(f["porc_min"]) - 0.01; aplicados.append(f"descuento ≥ {num(float(f['porc_min']), 0)}%")
    if f.get("porc_max") is not None:
        m &= df["porc"] <= float(f["porc_max"]) + 0.01; aplicados.append(f"descuento ≤ {num(float(f['porc_max']), 0)}%")
    d = df[m]

    metric = spec.get("metrica") if spec.get("metrica") in METRICS else "exonerado"
    dims = [x for x in (spec.get("agrupar_por") or []) if x in DIMS][:3]
    keys = [DIMS[x] for x in dims]
    top = int(spec.get("top") or 15)
    temporal = any(x in ("anio", "mes") for x in dims)
    orden = spec.get("orden") or ("natural" if temporal else "desc")

    tot = _agg(d) if len(d) else {k: 0 for k in METRICS}
    resumen = {"registros": int(len(d)), "filtros": aplicados, "metrica": metric, "total": tot,
               "periodo": f"{mes(d['tiempo'].min())} a {mes(d['tiempo'].max())}" if len(d) else "—",
               "ultimo_mes": d["tiempo"].max() if len(d) else None}
    if d.empty or not keys:
        return pd.DataFrame(), resumen

    g = d.groupby(keys, observed=True).agg(exonerado=("exonerado", "sum"), monto_total=("monto_total", "sum"), pagado=("pagado", "sum"),
                                            prefacturas=("num_prefactura", "size"), pacientes=("pid", "nunique"))
    g = g.join(d.drop_duplicates(keys + ["pid", "fecha"]).groupby(keys, observed=True).size().rename("atenciones"))
    g["exonerado_por_paciente"] = g["exonerado"] / g["pacientes"].replace(0, np.nan)
    g["pct_exonerado"] = g["exonerado"] / g["monto_total"].replace(0, np.nan) * 100
    g = g[g["prefacturas"] > 0]
    if metric in ADITIVAS:
        s = g[metric].sum()
        g["pct"] = g[metric] / s * 100 if s else np.nan
    resumen["n_grupos"] = int(len(g))
    if orden == "natural":
        g = g.sort_index()
        g = g.tail(top) if (len(g) > top and temporal) else g.head(top)
    else:
        g = g.sort_values(metric, ascending=(orden == "asc")).head(top)
    g = g.reset_index()
    if "tiempo" in g.columns:
        g["tiempo"] = pd.to_datetime(g["tiempo"]).dt.strftime("%Y-%m")
    cols = keys + [metric] + (["pct"] if "pct" in g.columns else [])
    extra = [c for c in ("exonerado", "pacientes") if c != metric and c not in cols]
    g = g[cols + extra]
    g.columns = [DIM_LABEL[dims[keys.index(c)]] if c in keys else (METRICS.get(c) or ("% del total" if c == "pct" else c)) for c in g.columns]
    return g, resumen


def _fmt_val(v, metric):
    if metric in MONEY:
        return "S/ " + num(v, 0)
    if metric == "pct_exonerado":
        return num(v, 1) + "%"
    return num(v, 0)


def tabla_a_texto(tb: pd.DataFrame, resumen: dict) -> str:
    out = {"periodo": resumen["periodo"], "filtros_aplicados": resumen["filtros"], "registros": resumen["registros"],
           "total": {k: (None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)) for k, v in resumen["total"].items()},
           "grupos_totales": resumen.get("n_grupos", 0)}
    if not tb.empty:
        t = tb.head(MAX_ROWS_LLM).copy()
        for c in t.columns:
            if pd.api.types.is_float_dtype(t[c]):
                t[c] = t[c].astype("float64").round(2)
        out["filas"] = json.loads(t.to_json(orient="records", force_ascii=False))
    return json.dumps(out, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Modo IA
# ---------------------------------------------------------------------------
def _contexto(df: pd.DataFrame) -> str:
    def vals(c, n=60):
        return ", ".join(map(str, sorted(df[c].dropna().astype(str).unique().tolist())[:n]))
    return f"""Eres el asistente de datos del «Observatorio de Exoneraciones del Servicio Social» del Instituto Nacional de Enfermedades
Neoplásicas (INEN, Perú). Respondes en español, con frases breves y claras, sin guiones largos y usando dos puntos solo si es imprescindible.

Datos: prefacturas que recibieron descuento del Servicio Social, serie mensual {mes(df['tiempo'].min())} a {mes(df['tiempo'].max())}.
Monto total = valor de la prefactura a tarifa INEN. Exonerado = descuento otorgado (ingreso que el INEN deja de percibir). Pagado = total − exonerado.
% exonerado ponderado = exonerado / monto total. Atención = paciente distinto en un día. Paciente = seudónimo (no hay DNI ni nombre).
Los pacientes únicos no se suman entre filas. La fecha es la de última modificación de la prefactura. El último mes y el último año pueden estar incompletos.
La base solo contiene prefacturas con descuento: no permite saber cuántos pacientes evaluados no recibieron descuento.

Valores válidos:
- Tipo de historia: {vals('tipo_hc')}
- Condición (código COD_CONDICION_SS): {vals('condicion')}
- Tramo de descuento: {vals('tramo')} · % de descuento (escala): {vals('porc_escala')}
- Ámbito territorial: {vals('ambito_geo')}
- Departamentos: {vals('departamento', 30)}

Reglas:
1. Toda cifra debe salir de la herramienta consultar_datos; nunca la inventes. Si la pregunta no se puede responder con estos datos, dilo.
2. Para variaciones pide la serie y calcula la diferencia, indicando qué años comparas; evita comparar un año incompleto con uno completo.
3. Montos en soles con separador de miles por espacio y coma decimal (S/ 1 234 567). Porcentajes con un decimal.
4. Si hablas de asistentes sociales, recuerda que las diferencias pueden reflejar el tipo de pacientes que atienden.
5. No muestres JSON ni nombres de parámetros. Responde en 2 a 6 frases o una lista corta; la tabla se muestra aparte.
"""


def _api_key() -> str | None:
    k = st.session_state.get("asist_key") or None
    if not k:
        try:
            k = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:  # noqa: BLE001
            k = None
    return k or os.environ.get("ANTHROPIC_API_KEY")


def _modelo() -> str:
    try:
        return st.secrets.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    except Exception:  # noqa: BLE001
        return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


def responder_llm(df: pd.DataFrame, pregunta: str, historial: list[dict], key: str):
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    msgs = []
    for h in historial[-6:]:
        msgs.append({"role": "user", "content": h["q"]})
        msgs.append({"role": "assistant", "content": h["a"]})
    msgs.append({"role": "user", "content": pregunta})
    ultima_tabla, ultimo_spec = pd.DataFrame(), None
    for _ in range(MAX_TOOL_ROUNDS):
        r = client.messages.create(model=_modelo(), max_tokens=1200, system=_contexto(df), tools=[TOOL], messages=msgs)
        if r.stop_reason != "tool_use":
            texto = "".join(b.text for b in r.content if b.type == "text").strip()
            return texto or "No obtuve respuesta.", ultima_tabla, ultimo_spec
        msgs.append({"role": "assistant", "content": r.content})
        res = []
        for b in r.content:
            if b.type == "tool_use":
                try:
                    tb, rs = consultar(df, b.input or {})
                    if not tb.empty:
                        ultima_tabla, ultimo_spec = tb, b.input
                    res.append({"type": "tool_result", "tool_use_id": b.id, "content": tabla_a_texto(tb, rs)})
                except Exception as e:  # noqa: BLE001
                    res.append({"type": "tool_result", "tool_use_id": b.id, "content": f"Error: {e}", "is_error": True})
        msgs.append({"role": "user", "content": res})
    return "La consulta requirió demasiados pasos; intenta formularla de forma más específica.", ultima_tabla, ultimo_spec


def reformular_llm(df: pd.DataFrame, q: str, key: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    instr = ("Tarea: el usuario escribió una pregunta para consultar estos datos. Propón UNA versión más precisa y respondible "
             "(métrica, periodo, grupo y desglose cuando tenga sentido), sin cambiar su intención. Si pide causas o proyecciones, "
             'reorienta hacia lo que los datos sí muestran. Responde solo con JSON: {"pregunta": "...", "motivo": "una frase breve"}')
    r = client.messages.create(model=_modelo(), max_tokens=300, system=_contexto(df) + "\n" + instr, messages=[{"role": "user", "content": q}])
    txt = "".join(b.text for b in r.content if b.type == "text")
    mm = re.search(r"\{.*\}", txt, re.S)
    out = json.loads(mm.group(0)) if mm else {}
    return {"pregunta": str(out.get("pregunta", "")).strip(), "motivo": str(out.get("motivo", "")).strip()}


# ---------------------------------------------------------------------------
# Modo básico (reglas)
# ---------------------------------------------------------------------------
_DIM_WORDS = [
    (r"por mes|mensual|cada mes", "mes"), (r"por ano|anual|cada ano|por anos|evolucion|tendencia", "anio"),
    (r"asistente|trabajador(a)? social|por personal", "asistente"),
    (r"por tramo|tramos", "tramo"), (r"por porcentaje|escala|por % ", "porcentaje"),
    (r"por condicion|segun condicion|cada condicion", "condicion"),
    (r"tipo de historia|por tipo", "tipo_historia"),
    (r"distrito", "distrito"), (r"provincia", "provincia"), (r"departamento|region", "departamento"),
    (r"ambito|lima metropolitana|territori", "ambito_territorial"), (r"documento|dni", "documento"),
]


def interpretar(df: pd.DataFrame, q: str) -> dict:
    n = _norm(q)
    f: dict = {}
    anios = sorted({int(a) for a in re.findall(r"\b(20[12]\d)\b", n)})
    if len(anios) == 1 and not re.search(r"desde|a partir", n):
        f["anio_desde"] = f["anio_hasta"] = anios[0]
    elif len(anios) == 1:
        f["anio_desde"] = anios[0]
    elif len(anios) >= 2:
        f["anio_desde"], f["anio_hasta"] = anios[0], anios[-1]
    conds = [c for c in df["condicion"].astype(str).unique() if c not in ("Sin dato",) and re.search(rf"\b{re.escape(_norm(c))}\b", n)]
    if conds and not re.search(r"por condicion|segun condicion|cada condicion", n):
        f["condicion"] = conds
    th = [t for t, pat in (("Historia clínica", r"historia clinica"), ("Externo", r"\bexterno"), ("Prevención", r"prevencion")) if re.search(pat, n)]
    if th:
        f["tipo_historia"] = th
    mp = re.search(r"(\d{2})\s*% a mas|(?:mayor|mas) de (\d{2})\s*%|(\d{2})\s*% o mas|>=?\s*(\d{2})", n)
    if mp:
        f["porc_min"] = float(next(x for x in mp.groups() if x))
    elif re.search(r"\b(\d{2})\s*%", n) and not re.search(r"por porcentaje", n):
        p = re.search(r"\b(\d{2})\s*%", n).group(1)
        f["porcentaje"] = [f"{p}%"]
    if re.search(r"lima metropolitana", n):
        f["ambito_territorial"] = ["Lima Metropolitana"]
    elif re.search(r"\bcallao\b", n) and "distrito" not in n:
        f["ambito_territorial"] = ["Callao"]
    elif re.search(r"provincias|fuera de lima|otras regiones", n):
        f["ambito_territorial"] = ["Otras regiones", "Lima provincias"]
    elif re.search(r"extranjer", n):
        f["ambito_territorial"] = ["Extranjero"]
    deps = [d for d in df["departamento"].dropna().astype(str).unique() if len(d) > 3 and d not in ("Lima", "Sin ubigeo", "Extranjero")
            and re.search(rf"\b{re.escape(_norm(d))}\b", n)]
    if deps:
        f["departamento"] = deps
    dist = [d for d in df["distrito"].dropna().astype(str).unique() if len(d) > 4 and d not in ("Sin dato", "Lima", "Callao")
            and re.search(rf"\b{re.escape(_norm(d))}\b", n)]
    dist = [x for x in dist if x not in f.get("departamento", [])]
    if dist:
        f["distrito"] = dist
    asis = [a for a in df["asistente"].dropna().astype(str).unique() if a != "Sin asistente registrado"
            and (re.search(rf"\b{re.escape(_norm(a))}\b", n) or (len(_norm(a).split()) >= 2 and " ".join(_norm(a).split()[:2]) in n))]
    if asis:
        f["asistente"] = asis
    # métrica
    if re.search(r"por paciente|promedio por paciente", n):
        metrica = "exonerado_por_paciente"
    elif re.search(r"% exonerado|porcentaje exonerado|porcentaje de descuento promedio|descuento promedio|tasa de descuento", n):
        metrica = "pct_exonerado"
    elif re.search(r"cuantos pacientes|numero de pacientes|pacientes unicos|beneficiad|\bpacientes\b", n) and not re.search(r"monto|exonerad|soles", n):
        metrica = "pacientes"
    elif re.search(r"prefactura|cuantas|numero de casos|casos", n) and not re.search(r"monto|soles", n):
        metrica = "prefacturas"
    elif re.search(r"atenciones", n):
        metrica = "atenciones"
    elif re.search(r"pagad|pago del paciente|copago", n):
        metrica = "pagado"
    elif re.search(r"monto total|facturad|prefacturad", n):
        metrica = "monto_total"
    else:
        metrica = "exonerado"
    dims = []
    for pat, d in _DIM_WORDS:
        if re.search(pat, n) and d not in dims:
            if d == "asistente" and f.get("asistente") and not re.search(r"por asistente|cada asistente|asistentes", n):
                continue
            if d == "anio" and "mes" in dims:
                continue
            if d == "ambito_territorial" and f.get("ambito_territorial"):
                continue
            if d == "tipo_historia" and f.get("tipo_historia"):
                continue
            if d == "documento" and not re.search(r"por documento|tipo de documento|sin dni", n):
                continue
            if d == "departamento" and (f.get("departamento") or "distrito" in dims or "provincia" in dims):
                continue
            dims.append(d)
    mtop = re.search(r"\btop\s*(\d{1,3})|\b(\d{1,3})\s+(?:principales|primer|mayores|distritos|asistentes|departamentos)", n)
    top = int(next(x for x in mtop.groups() if x)) if mtop else (150 if "mes" in dims else 15)
    spec = {"metrica": metrica, "agrupar_por": dims[:2], "top": top, "filtros": f}
    if re.search(r"menor|menos", n) and not re.search(r"al menos", n):
        spec["orden"] = "asc"
    return spec


def describir(tb: pd.DataFrame, rs: dict) -> str:
    met = rs["metrica"]
    tot = rs["total"].get(met, np.nan)
    partes = [f"**{METRICS[met]}** · {rs['periodo']}"]
    if rs["filtros"]:
        partes.append("Filtros: " + "; ".join(rs["filtros"]) + ".")
    if rs["registros"] == 0:
        return "No hay prefacturas con esos criterios. Prueba con menos filtros o revisa los del panel."
    if met == "pacientes" and not tb.empty:
        partes.append(f"Total: **{num(tot)}** pacientes únicos (no se suman entre filas).")
    else:
        partes.append(f"Total: **{_fmt_val(tot, met)}** ({num(rs['total']['prefacturas'])} prefacturas, {num(rs['total']['pacientes'])} pacientes).")
    if not tb.empty:
        dcols = [c for c in tb.columns if c in DIM_LABEL.values()]
        vcol = tb.columns[len(dcols)]
        first = tb.iloc[0]
        if dcols and not any(c in ("Año", "Mes") for c in dcols):
            lbl = " · ".join(str(first[c]) for c in dcols)
            extra = f" ({num(first['% del total'], 1)}% del total)" if "% del total" in tb.columns else ""
            partes.append(f"Primer lugar: **{lbl}** con {_fmt_val(first[vcol], met)}{extra}.")
        elif dcols == ["Año"] and len(tb) >= 2:
            a, b = tb.iloc[-2], tb.iloc[-1]
            um = rs.get("ultimo_mes")
            if um is not None and int(b["Año"]) == um.year and um.month < 12 and len(tb) >= 3:
                a, b = tb.iloc[-3], tb.iloc[-2]
                partes.append(f"{int(tb.iloc[-1]['Año'])} está incompleto (hasta {mes(um)}); se comparan los dos últimos años completos.")
            if a[vcol]:
                partes.append(f"Entre {int(a['Año'])} y {int(b['Año'])} la variación es de {num((b[vcol] / a[vcol] - 1) * 100, 1)}%.")
    partes.append("_Modo básico: para preguntas libres activa el modo IA con una clave de Anthropic._")
    return "  \n".join(partes)


# ---------------------------------------------------------------------------
# Evaluación de la pregunta
# ---------------------------------------------------------------------------
_RX_METRICA = r"exonera|descuento|monto|soles|pacientes|prefactura|atenciones|pagad|casos|cuant"
_RX_PERIODO = r"\b20[12]\d\b|por mes|mensual|por ano|anual|evolucion|tendencia|ultimo ano|este ano"
_RX_GRUPO = r"hos\b|ase\b|condicion|historia|externo|prevencion|tramo|%|asistente|lima|callao|distrito|departamento|region|provincia|extranjer"
_RX_FUERA = r"por que|porque|causa|motivo|razon|predic|proyecc|proyect|pronostic|futuro|2027|2028|2029|2030"


def _insertar(q: str, frag: str) -> str:
    q = q.rstrip()
    m = re.search(r"[?.!]+$", q)
    return (q[:m.start()] + frag + m.group(0)) if m else (q + frag)


def evaluar(df: pd.DataFrame, q: str, spec: dict | None = None) -> dict:
    n = _norm(q)
    spec = spec or interpretar(df, q)
    f, dims = spec.get("filtros", {}), spec.get("agrupar_por", [])
    um = df["tiempo"].max() if len(df) else pd.Timestamp("2026-01-01")
    anio = int(um.year if um.month == 12 else um.year - 1)
    elem = [
        ("metrica", "Métrica", bool(re.search(_RX_METRICA, n)), " (monto exonerado)", "Qué medir: monto exonerado, pacientes, prefacturas o exonerado por paciente."),
        ("periodo", "Periodo", bool(f.get("anio_desde") or f.get("mes_desde") or re.search(_RX_PERIODO, n)), f" en {anio}", "Cuándo: un año, un rango o la evolución."),
        ("grupo", "Grupo", bool(any(f.get(k) for k in _FILTROS_LISTA) or f.get("porc_min") or re.search(_RX_GRUPO, n)), " en la condición HOS",
         "A quién: una condición, tipo de historia, tramo, asistente o zona."),
        ("desglose", "Desglose", bool(dims), " por tramo de descuento", "Cómo abrir el resultado: por año, tramo, asistente, distrito…"),
    ]
    faltan = [e for e in elem if not e[2]]
    avisos = []
    anios = {int(a) for a in re.findall(r"\b(20[12]\d)\b", n)}
    if len(df) and anios:
        lo, hi = int(df["tiempo"].min().year), int(df["tiempo"].max().year)
        fuera = sorted(a for a in anios if a < lo or a > hi)
        if fuera:
            avisos.append(f"No hay datos para {', '.join(map(str, fuera))}; la serie cubre {lo} a {hi}.")
        if um.year in anios and um.month < 12:
            avisos.append(f"{um.year} está incompleto (datos hasta {mes(um)}); para comparar años usa años completos.")
    if re.search(_RX_FUERA, n):
        avisos.append("Los datos describen los descuentos otorgados; no explican causas ni proyectan. Revisa las pestañas Tendencias y Proyecciones.")
    partes = [METRICS.get(spec.get("metrica", "exonerado"), "Monto exonerado")]
    if f.get("anio_desde") or f.get("anio_hasta"):
        a, b = f.get("anio_desde"), f.get("anio_hasta")
        partes.append(str(a) if a == b else f"{a or '…'} a {b or '…'}")
    for k in _FILTROS_LISTA:
        if f.get(k):
            partes.append(", ".join(map(str, f[k][:2])) + ("…" if len(f[k]) > 2 else ""))
    if f.get("porc_min"):
        partes.append(f"descuento ≥ {num(f['porc_min'], 0)}%")
    if dims:
        partes.append("por " + " y ".join(DIM_LABEL[d].lower() for d in dims))
    n_ok = len(elem) - len(faltan)
    nivel = "ok" if n_ok >= 3 else ("medio" if n_ok >= 2 else "bajo")
    return {"nivel": nivel, "faltan": faltan, "avisos": avisos, "lectura": " · ".join(partes)}


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
EJEMPLOS = [
    "Evolución anual del monto exonerado",
    "Monto exonerado por tramo de descuento en 2025",
    "¿Cuántos pacientes por condición en 2025?",
    "Top 10 distritos de Lima Metropolitana por monto exonerado",
    "Exonerado por paciente según tipo de historia",
    "Prefacturas con 90% a más por asistente en 2025",
]
_BTN_TEXTO = {"type": "tertiary"} if _V >= (1, 42) else {}
_DOT = {"ok": ("#2F9E44", "Pregunta precisa. Puedes consultar."),
        "medio": ("#E8A33D", "Las preguntas más precisas funcionan mejor. Prueba añadir:"),
        "bajo": ("#C94141", "Pregunta muy general. Prueba añadir:")}


def _caja(key: str, **kw):
    try:
        return st.container(key=key, **kw)
    except TypeError:
        return st.container(**kw)


def _cb_enviar():
    q = (st.session_state.get("asist_q") or "").strip()
    if q:
        st.session_state["asist_run"] = q
    st.session_state["asist_q"] = ""
    st.session_state.pop("asist_sug", None)


def _cb_limpiar():
    st.session_state["asist_hist"] = []
    st.session_state["asist_q"] = ""
    st.session_state.pop("asist_sug", None)


def _cb_ejemplo(txt: str):
    st.session_state["asist_run"] = txt
    st.session_state["asist_q"] = ""
    st.session_state.pop("asist_sug", None)


def _cb_chip(frags: dict):
    sel = st.session_state.get("asist_chip")
    if sel and sel in frags:
        st.session_state["asist_q"] = _insertar(st.session_state.get("asist_q") or "", frags[sel])
    st.session_state["asist_chip"] = None


def _cb_usar_sug():
    sug = st.session_state.pop("asist_sug", None)
    if sug and sug.get("pregunta"):
        st.session_state["asist_q"] = sug["pregunta"]


def _cb_cerrar():
    st.session_state["asist_oculto"] = True


def _panel_evaluacion(base: pd.DataFrame, q: str, key: str | None):
    ev = evaluar(base, q)
    color, msg = _DOT[ev["nivel"]]
    st.markdown(f"<div class='asist-eval'><span class='asist-dot' style='background:{color}'></span>{msg}</div>", unsafe_allow_html=True)
    if ev["faltan"]:
        frags = {e[1]: e[3] for e in ev["faltan"]}
        st.pills("Elementos sugeridos", list(frags), selection_mode="single", key="asist_chip", label_visibility="collapsed",
                 on_change=_cb_chip, args=(frags,), help="Al tocar uno se añade un ejemplo editable a tu pregunta. "
                 + " ".join(f"{e[1]}: {e[4]}" for e in ev["faltan"]))
    for a in ev["avisos"]:
        st.markdown(f"<div class='asist-aviso'>⚠️ {a}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='asist-lectura'>{'Lectura preliminar' if key else 'Así la entiendo'}: {ev['lectura']}</div>", unsafe_allow_html=True)
    if key and ev["nivel"] != "ok":
        if st.button("✨ Mejorar la pregunta con IA", key="asist_refine", **STRETCH):
            with st.spinner("Reformulando…"):
                try:
                    st.session_state["asist_sug"] = {**reformular_llm(base, q, key), "origen": q}
                except Exception as e:  # noqa: BLE001
                    st.session_state["asist_sug"] = {"pregunta": "", "motivo": f"No se pudo reformular: {e}", "origen": q}
    sug = st.session_state.get("asist_sug")
    if sug and sug.get("origen") == q:
        if sug.get("pregunta"):
            st.markdown(f"<div class='asist-sug'>💡 {sug['pregunta']}<br><span>{sug.get('motivo', '')}</span></div>", unsafe_allow_html=True)
            st.button("Usar esta versión", key="asist_usar", on_click=_cb_usar_sug, **STRETCH)
        else:
            st.caption(sug.get("motivo", ""))


def render_input(df_filtrado: pd.DataFrame, df_total: pd.DataFrame) -> None:
    """Barra lateral: caja de la pregunta, ejemplos y modo. Ejecuta la consulta y la guarda en el historial."""
    st.session_state.setdefault("asist_hist", [])
    st.session_state.setdefault("asist_q", "")
    with st.expander("💬 Pregunta a los datos", expanded=True):
        key = _api_key()
        usar_flt = st.toggle("Aplicar los filtros del panel", value=True, key="asist_flt",
                             help="Si está activo, el asistente responde sobre la selección actual (meses, condición, tramo, territorio, asistente).")
        base = df_filtrado if usar_flt else df_total
        st.text_area("Tu pregunta", key="asist_q", height=80, label_visibility="collapsed",
                     placeholder="Ej.: ¿cuánto se exoneró por tramo de descuento en 2025?",
                     help="Pulsa Ctrl+Enter para revisar la pregunta antes de consultar.")
        q_actual = (st.session_state.get("asist_q") or "").strip()
        if q_actual and len(base):
            _panel_evaluacion(base, q_actual, key)
        c1, c2 = st.columns(2)
        c1.button("Consultar", type="primary", on_click=_cb_enviar, key="asist_send", **STRETCH)
        c2.button("Limpiar", on_click=_cb_limpiar, key="asist_clear", **STRETCH)

        pregunta = st.session_state.pop("asist_run", None)
        if pregunta:
            with st.spinner("Consultando…"):
                try:
                    if key:
                        a, tb, spec = responder_llm(base, pregunta, st.session_state["asist_hist"], key)
                        modo = "IA"
                    else:
                        spec = interpretar(base, pregunta)
                        tb, rs = consultar(base, spec)
                        a, modo = describir(tb, rs), "básico"
                except Exception as e:  # noqa: BLE001
                    a, tb, spec, modo = f"No se pudo completar la consulta: {e}", pd.DataFrame(), None, "error"
            st.session_state["asist_hist"].append({"q": pregunta, "a": a, "tabla": tb, "spec": spec, "modo": modo})
            st.session_state["asist_oculto"] = False

        st.markdown("<div class='asist-ej-tit'>Ejemplos de preguntas (tócalos para consultar)</div>", unsafe_allow_html=True)
        with _caja("asist_ejs"):
            for i, ej in enumerate(EJEMPLOS):
                st.button(ej, key=f"asist_ej_{i}", on_click=_cb_ejemplo, args=(ej,), **_BTN_TEXTO)
        if st.session_state["asist_hist"]:
            st.caption("El resultado se muestra en la sección principal, arriba de las pestañas.")
        st.caption(("🟢 Modo IA activo (Claude). " if key else "⚪ Modo básico por reglas. ")
                   + "Las cifras salen de agregaciones sobre los datos; al modelo solo se envían tablas resumidas, sin identificadores.")
        if not key:
            with st.popover("Activar modo IA", **STRETCH):
                st.text_input("Clave de API de Anthropic", type="password", key="asist_key",
                              help="Se guarda solo en esta sesión. En Streamlit Cloud, colócala en Secrets como ANTHROPIC_API_KEY.")


def _grafico(tb: pd.DataFrame, height: int = 340):
    dcols = [c for c in tb.columns if c in DIM_LABEL.values()]
    if not dcols or len(tb) < 2:
        return
    import plotly.graph_objects as go
    vcol = tb.columns[len(dcols)]
    money = "(S/)" in vcol
    if len(dcols) == 2 and dcols[0] in ("Año", "Mes"):
        fig = go.Figure()
        from ui import PALETTE
        for i, (k, x) in enumerate(tb.groupby(dcols[1], observed=True, sort=False)):
            fig.add_trace(go.Bar(x=x[dcols[0]].astype(str), y=x[vcol], name=str(k), marker_color=PALETTE[i % len(PALETTE)]))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(type="category")
    elif dcols[0] in ("Año", "Mes"):
        x = tb[dcols[0]].astype(str)
        fig = go.Figure(go.Scatter(x=x, y=tb[vcol], mode="lines+markers", line=dict(color=PRIMARY, width=2.5), marker=dict(size=7)))
        fig.update_xaxes(type="category")
    else:
        t = tb.head(20).iloc[::-1]
        lab = t[dcols].astype(str).agg(" · ".join, axis=1).str.slice(0, 45)
        fig = go.Figure(go.Bar(x=t[vcol], y=lab, orientation="h", marker_color=PRIMARY))
        fig.update_yaxes(type="category", tickfont=dict(size=11))
        height = max(height, 26 * len(t) + 60)
    fig.update_layout(height=height, margin=dict(l=0, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=len(dcols) == 2, hovermode="closest", legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(gridcolor="#eceff4")
    fig.update_yaxes(gridcolor="#eceff4")
    if money:
        fig.update_xaxes(tickprefix="S/ ", tickformat=",.0f") if dcols[0] not in ("Año", "Mes") else fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
    st.plotly_chart(fig, config={"displayModeBar": False}, **STRETCH)


def _tabla(tb: pd.DataFrame, height: int | None = None):
    fm = {}
    for c in tb.columns:
        if pd.api.types.is_numeric_dtype(tb[c]) and c != "Año":
            if "(S/)" in c:
                fm[c] = lambda v: "S/ " + num(v, 0)
            elif c.startswith("%"):
                fm[c] = lambda v: num(v, 1) + "%"
            else:
                fm[c] = lambda v: num(v, 0)
    kw = {"height": int(height)} if height else {}
    st.dataframe(tb.style.format(fm, na_rep="—"), hide_index=True, **STRETCH, **kw)


def render_resultado() -> bool:
    """Sección principal: recuadro con el último resultado del asistente. Devuelve True si se dibujó."""
    hist = st.session_state.get("asist_hist") or []
    if not hist or st.session_state.get("asist_oculto"):
        return False
    i = len(hist) - 1
    h = hist[i]
    with _caja("asist_main", border=True):
        c1, c2 = st.columns([6, 1])
        c1.markdown(f"<div class='asist-q'>🧑 {h['q']}</div>", unsafe_allow_html=True)
        c2.button("✕ Cerrar", key="asist_close", on_click=_cb_cerrar, **STRETCH)
        tb = h["tabla"]
        if tb is not None and not tb.empty:
            a1, a2 = st.columns([1, 1.5])
            with a1:
                st.markdown(h["a"])
            with a2:
                _grafico(tb)
            with st.expander(f"Tabla de resultados ({len(tb)} filas)", expanded=False):
                _tabla(tb, min(420, 38 * len(tb) + 40))
            b = st.columns(3)
            b[0].download_button("⬇️ Descargar CSV", tb.to_csv(index=False).encode("utf-8-sig"), file_name="consulta_servicio_social.csv",
                                 mime="text/csv", key=f"asist_dl_{i}", **STRETCH)
        else:
            st.markdown(h["a"])
            b = st.columns(3)
        if h.get("spec"):
            with b[1].popover("Cómo se calculó", **STRETCH):
                st.json(h["spec"], expanded=True)
        if len(hist) > 1:
            with b[2].popover(f"Historial ({len(hist) - 1})", **STRETCH):
                for j in range(len(hist) - 2, -1, -1):
                    st.markdown(f"**{hist[j]['q']}**")
                    st.markdown(hist[j]["a"])
                    st.divider()
    st.divider()
    return True
