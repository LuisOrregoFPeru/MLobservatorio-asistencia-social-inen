# Observatorio de Exoneraciones del Servicio Social | INEN

Dashboard Streamlit sobre las prefacturas con descuento autorizado por el Servicio Social del INEN (ene-2016 a oct-2026).
Misma arquitectura visual que el Observatorio del Consumo por IAFAS (`ui.py` y `stats_analysis.py` son los mismos módulos).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pestañas: Resumen · Tendencias · Nivel de exoneración · Pacientes · Territorio · Asistentes sociales · Control y alertas · Proyecciones · Tabla dinámica · Metodología.

Datos: `data/fact_ss.parquet` (seudonimizado: sin nombres, DNI, historia clínica ni dirección; se conserva el número de prefactura) y
`data/peru_departamentos.geojson` (límites departamentales para el mapa). Para regenerar desde un nuevo Excel:
`python build_data.py RUTA/DATA_SERVICIO_SOCIAL.xlsx` (la sal del seudónimo se cambia con la variable de entorno `PID_SALT`).

Archivos: `app.py` (navegación, filtros, tabla dinámica, metodología), `panels.py` (paneles), `data_processing.py` (carga y agregaciones),
`build_data.py` (Excel → parquet), `ui.py` (estilos y formato es-PE), `stats_analysis.py` (Mann-Kendall, APC, Chow, Holt).

### Asistente «Pregunta a los datos»

La pregunta se escribe en la barra lateral (arriba de los filtros) y el resultado aparece en la sección principal, en un recuadro sobre las pestañas,
con texto, gráfico, tabla desplegable, descarga CSV, «Cómo se calculó» e historial; debajo lleva una línea de separación. «Cerrar» oculta el recuadro.

- **Modo IA (Claude):** se activa con `ANTHROPIC_API_KEY` en `.streamlit/secrets.toml` (ver `secrets.toml.example`), en Secrets de Streamlit Cloud
  o ingresando la clave en la barra lateral. El modelo no ejecuta código: llama a la herramienta `consultar_datos` (agregaciones predefinidas en
  `asistente.py`) y solo recibe tablas resumidas, sin identificadores.
- **Modo básico (sin clave):** intérprete por reglas que reconoce año, condición, tipo de historia, tramo o % de descuento, asistente, territorio,
  métrica, desglose y ranking.
