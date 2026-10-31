"""
build_data.py
-------------
Convierte la extracción de prefacturas con descuento del Servicio Social del INEN
(Excel, hoja «data», consulta sobre INEN.HOS_PRE_FACTURA_CAB_TMP) en un archivo
compacto y SEUDONIMIZADO:

  data/fact_ss.parquet   una fila por prefactura con descuento (VAL_MONTO_DSCTO > 0)

Uso:
    python build_data.py RUTA/DATA_SERVICIO_SOCIAL.xlsx

No se guardan nombres, DNI, historia clínica ni dirección. El paciente se reemplaza
por un entero derivado de un hash con sal (PID_SALT) de tipo de historia + número de
historia, suficiente para contar pacientes únicos y seguir su recurrencia. Se conserva
el número de prefactura para que el auditor pueda rastrear un caso en el sistema.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"
PID_SALT = os.environ.get("PID_SALT", "ss-dashboard-v1")

# Porcentajes de descuento que aparecen como escala habitual (≥ 0,5% de las prefacturas)
ESCALA = [20, 25, 30, 35, 40, 50, 60, 90]
TRAMO_BINS = [0, 20, 30, 40, 59.99, 89.99, 101]
TRAMO_LABELS = ["Hasta 20%", "21-30%", "31-40%", "41-59%", "60-89%", "90% a más"]
EXTRANJERO = {"AMERICA", "EUROPA", "ASIA", "AFRICA", "OCEANIA"}


def _h(x: str) -> int:
    return int(hashlib.sha256((PID_SALT + x).encode()).hexdigest()[:12], 16)


def parse_ubigeo(s: pd.Series) -> pd.DataFrame:
    """'LIMA-LIMA-COMAS' -> departamento, provincia, distrito. Corrige el rótulo 'CALLAO (PRO' del origen."""
    u = s.fillna("").astype(str).str.strip()
    u = u.str.replace(r"^CALLAO\s*\(PRO", "CALLAO", regex=True)
    parts = u.str.split("-", n=2, expand=True).reindex(columns=[0, 1, 2])
    parts = parts.apply(lambda c: c.fillna("").str.strip())
    dep, prov, dist = parts[0], parts[1], parts[2]
    tt = lambda x: x.str.title().str.replace(r"\b(De|Del|La|Las|Los|Y)\b", lambda m: m.group(0).lower(), regex=True).str.replace(r"^(de|del|la|las|los)\b", lambda m: m.group(0).title(), regex=True)
    out = pd.DataFrame({"departamento": tt(dep), "provincia": tt(prov), "distrito": tt(dist)})
    out.loc[dep == "", "departamento"] = "Sin ubigeo"
    ext = dep.isin(EXTRANJERO)
    out.loc[ext, "departamento"] = "Extranjero"
    out.loc[ext, "provincia"] = dep[ext].str.title()
    for c in ("provincia", "distrito"):
        out.loc[out[c] == "", c] = "Sin dato"
    # ámbito territorial
    amb = np.select(
        [dep == "", ext, (dep == "LIMA") & (prov == "LIMA"), dep == "CALLAO", (dep == "LIMA") & (prov == ""), dep == "LIMA"],
        ["Sin ubigeo", "Extranjero", "Lima Metropolitana", "Callao", "Lima (sin provincia)", "Lima provincias"],
        default="Otras regiones",
    )
    out["ambito_geo"] = amb
    return out


def dni_estado(s: pd.Series) -> pd.Series:
    d = s.fillna("").astype(str).str.strip()
    return pd.Series(np.select(
        [d == "", d.str.contains("INDOC", case=False), d.str.fullmatch(r"\d{8}")],
        ["Sin documento", "Indocumentado / extranjero", "DNI"], default="Otro documento"), index=s.index)


def main(path: str) -> None:
    raw = pd.read_excel(path, sheet_name="data", dtype=str)
    print("filas crudas:", len(raw))
    df = pd.DataFrame()
    df["num_prefactura"] = raw["NUM_PREFACTURA"].str.strip()
    df["fecha"] = pd.to_datetime(raw["FECHA"]).dt.normalize()
    df["tiempo"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    df["anio"] = df["fecha"].dt.year.astype("int16")
    key = raw["TIPO_HC"].fillna("") + "|" + raw["HISTORIA_PACIENTE"].fillna("").str.strip()
    df["pid"] = key.map(_h).astype("int64")
    df["tipo_hc"] = raw["TIPO_HC"].fillna("Sin dato").str.strip().str.capitalize().replace({"Prevención": "Prevención", "Historia clinica": "Historia clínica"})
    df["condicion"] = raw["CONDICION"].fillna("Sin dato").str.strip()
    df["responsable"] = raw["COD_RESPONSABLE_CTA"].fillna("Sin dato").str.strip()
    df["doc_estado"] = dni_estado(raw["DNI"])
    df["monto_total"] = pd.to_numeric(raw["VAL_MONTO_TOTAL"], errors="coerce").fillna(0.0)
    df["exonerado"] = pd.to_numeric(raw["VAL_MONTO_DSCTO"], errors="coerce").fillna(0.0)
    df["pagado"] = pd.to_numeric(raw["T_PAGO"], errors="coerce").fillna(df["monto_total"] - df["exonerado"])
    df["porc"] = pd.to_numeric(raw["T_DSCTO_PORC"], errors="coerce")
    df["porc_entero"] = df["porc"].round(0)
    df["porc_escala"] = np.where(np.isclose(df["porc"], df["porc_entero"]) & df["porc_entero"].isin(ESCALA),
                                 df["porc_entero"].astype("Int64").astype(str) + "%", "Fuera de escala")
    df["tramo"] = pd.cut(df["porc"], bins=TRAMO_BINS, labels=TRAMO_LABELS, include_lowest=True).astype(str)
    asis = raw["ASISTENTE_SOCIAL"].fillna("").str.strip()
    df["asistente"] = asis.replace("", "Sin asistente registrado").str.title()
    geo = parse_ubigeo(raw["DESC_UBIGEO"])
    df = pd.concat([df, geo], axis=1)

    # número de prefacturas del mismo paciente en el mismo día (atención = paciente-día)
    df["n_pref_dia"] = df.groupby(["pid", "fecha"])["num_prefactura"].transform("size").astype("int16")
    df["porc_distintos_dia"] = df.groupby(["pid", "fecha"])["porc_entero"].transform("nunique").astype("int8")

    for c in ["tipo_hc", "condicion", "responsable", "doc_estado", "porc_escala", "tramo", "asistente",
              "departamento", "provincia", "distrito", "ambito_geo"]:
        df[c] = df[c].astype("category")
    for c in ["monto_total", "exonerado", "pagado", "porc", "porc_entero"]:
        df[c] = df[c].astype("float64")

    chk = (df["monto_total"] - df["exonerado"] - df["pagado"]).abs().max()
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / "fact_ss.parquet", index=False, compression="zstd")
    print("fact:", df.shape, "| pacientes:", df.pid.nunique(), "| máx. descuadre total-exonerado-pagado:", round(chk, 4))
    print("monto total S/ {:,.0f} | exonerado S/ {:,.0f} | pagado S/ {:,.0f}".format(df.monto_total.sum(), df.exonerado.sum(), df.pagado.sum()))
    print(df.groupby("anio").exonerado.sum().round(0))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python build_data.py RUTA/DATA_SERVICIO_SOCIAL.xlsx")
    main(sys.argv[1])
