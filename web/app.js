/* =============================================================
   Previsión del Brent: lógica de la web.
   La web no calcula nada: solo lee los JSON de data/ (generados en Python).
   ============================================================= */
"use strict";

const estado = { h: 5, hTest: 5, ciclosVisibles: true, datos: null };

// ---------- Paleta (tema oscuro único) ----------
const C = {
  texto: "#ECEDEF", texto2: "#9A9CA5", texto3: "#6B6D76", borde: "#2A2B31",
  tarjeta: "#212227", tarjeta2: "#1B1C21",
  menta: "#8FD9C4", ambar: "#F2B84B", rosa: "#E7A9CF", azul: "#5AA9E6",
  ambarSuave: "rgba(242,184,75,0.16)", sombreado: "rgba(255,255,255,0.035)",
};

// ---------- Formato ----------
const fmt = (x, d = 2) => x == null || Number.isNaN(x) ? "—"
  : x.toLocaleString("es-ES", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtPct = (x, d = 1) => x == null ? "—" : (x > 0 ? "+" : "") + fmt(x, d) + " %";
const fmtFecha = (iso) => {
  if (!iso) return "—";
  const [a, m, d] = iso.slice(0, 10).split("-");
  return `${d}-${m}-${a}`;
};
const fmtFechaLarga = (iso) => new Date(iso.slice(0, 10) + "T12:00:00")
  .toLocaleDateString("es-ES", { day: "numeric", month: "long", year: "numeric" });
const NOMBRES = { rnn: "RNN simple", lstm: "LSTM", gru: "GRU", ingenuo: "Ingenuo" };
const PLAZOS = { 5: "una semana (5 días hábiles)", 63: "unos tres meses (63 días hábiles)", 252: "un año (252 días hábiles)" };
const CORTOS = { 5: "1 semana", 63: "3 meses", 252: "1 año" };
const CONFIG_PLOTLY = { responsive: true, displayModeBar: false };

function iconos() { if (window.lucide) lucide.createIcons({ attrs: { "aria-hidden": "true" } }); }

// ---------- Carga ----------
async function cargar(nombre) {
  const r = await fetch(`data/${nombre}.json`, { cache: "no-cache" });
  if (!r.ok) throw new Error(`No se pudo cargar ${nombre}.json`);
  return r.json();
}

async function iniciar() {
  iconos();
  try {
    const [historico, prediccion, comparativa, test, ciclos, meta] = await Promise.all(
      ["historico", "prediccion", "comparativa", "test", "ciclos", "meta"].map(cargar));
    estado.datos = { historico, prediccion, comparativa, test, ciclos, meta };
  } catch (e) {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<div class="tarjeta" style="margin-top:20px">No se han podido cargar los datos (${e.message}).
       Si abres el fichero directamente, sírvelo con <code>python -m http.server</code>.</div>`);
    return;
  }
  pintarCabecera();
  pintarCiclos();
  pintarTabla();
  pintarBarras();
  conectarControles();
  actualizarHorizonte();
  pintarTest();
  navegacionActiva();
  iconos();
}

// ---------- Cabecera ----------
function pintarCabecera() {
  const { meta } = estado.datos;
  document.getElementById("precio-actual").textContent = fmt(meta.precio_ultimo);
  document.getElementById("fecha-dato").textContent = fmtFechaLarga(meta.fecha_ultimo_dato);
  document.getElementById("fecha-entreno").textContent = fmtFecha(meta.fecha_entrenamiento);
  document.getElementById("fecha-gen").textContent = fmtFecha(meta.fecha_generacion);
  document.getElementById("enlace-actions").href = meta.url_actions;
}

// ---------- Predicción por horizonte ----------
function filaComp(modelo, h) {
  return estado.datos.comparativa.find((f) => f.modelo === modelo && f.horizonte === h);
}

function actualizarHorizonte() {
  const { prediccion } = estado.datos;
  const p = prediccion.horizontes.find((x) => x.h === estado.h);
  const [li, ls] = p.intervalo_80;
  const [ili, ils] = p.ingenuo.intervalo_80;

  document.getElementById("t-precio").textContent = `${fmt(p.precio_previsto)} $`;
  document.getElementById("t-fecha-obj").textContent = `hacia el ${fmtFecha(p.fecha_objetivo_aprox)}`;
  const tVar = document.getElementById("t-var");
  tVar.textContent = fmtPct(p.variacion_pct);
  tVar.className = "kpi__valor " + (p.variacion_pct >= 0 ? "sube" : "baja");
  const icoVar = document.getElementById("ico-var");
  icoVar.className = "icono " + (p.variacion_pct >= 0 ? "icono--menta" : "icono--rosa");
  icoVar.innerHTML = `<i data-lucide="${p.variacion_pct >= 0 ? "trending-up" : "trending-down"}"></i>`;
  document.getElementById("t-rango").textContent = `${fmt(li, 0)} – ${fmt(ls, 0)} $`;
  document.getElementById("t-ingenuo").textContent = `${fmt(p.ingenuo.precio_previsto)} $`;
  document.getElementById("t-ingenuo-rango").textContent =
    `ingenuo: precio sin cambios · rango ${fmt(ili, 0)} – ${fmt(ils, 0)} $`;

  // Texto: qué significa
  const sentido = p.variacion_pct >= 0 ? "por encima" : "por debajo";
  document.getElementById("x-significa").innerHTML =
    `Con los datos hasta el ${fmtFechaLarga(prediccion.fecha_datos)}, el modelo estima que dentro de
     ${PLAZOS[p.h]} el Brent costará unos <strong>${fmt(p.precio_previsto)} $</strong>, un
     ${fmt(Math.abs(p.variacion_pct))} % ${sentido} del precio actual. Pero lo importante es el rango:
     con errores como los que el modelo cometió en el pasado, el precio debería quedar
     <strong>entre ${fmt(li, 0)} y ${fmt(ls, 0)} $</strong> en 8 de cada 10 casos.`;

  // Texto: fiabilidad a partir de la comparativa
  const red = estado.datos.test.red;
  const c = filaComp(red, p.h);
  const pv = c.DM_p_valor;
  const signif = pv < 0.05;
  const mejora = 100 * c.mejora_vs_ingenuo_ensemble;
  let txt = `En la prueba con datos de 2023 a 2026, el error del modelo a este plazo fue
    ${mejora >= 0 ? "un " + fmt(mejora, 1) + " % menor" : "un " + fmt(-mejora, 1) + " % mayor"} que el de la
    referencia más simple (suponer que el precio no cambia), pero
    ${signif ? "la diferencia es estadísticamente significativa" :
      "<strong>la diferencia no es significativa</strong> (p-valor de Diebold-Mariano = " + fmt(pv, 2) +
      "): podría deberse al azar"}.`;
  if (p.h === 252) {
    txt += ` A un año los resultados son además muy inestables: según la semilla de entrenamiento, el modelo
      quedaba en promedio un ${fmt(-100 * c.mejora_vs_ingenuo_media, 0)} % por detrás de la referencia. <strong>Lo útil
      aquí es el rango; la cifra central debe tomarse con mucha cautela.</strong>`;
  } else {
    txt += ` Úsalo como una referencia más, no como una certeza.`;
  }
  txt += ` En la prueba, el precio real cayó dentro del rango del 80 % el ${fmt(c["cobertura_80_%_ensemble"], 0)} % de las veces.`;
  document.getElementById("x-fiabilidad").innerHTML = txt;

  pintarPrincipal();
  pintarGauge();
  iconos();
}

// ---------- Estilo común de Plotly ----------
const FORMATO_FECHAS = [
  { dtickrange: [null, "M1"], value: "%d-%m-%Y" },
  { dtickrange: ["M1", "M12"], value: "%m-%Y" },
  { dtickrange: ["M12", null], value: "%Y" },
];

const estrecho = () => window.innerWidth < 600;

function layoutBase() {
  const movil = estrecho();
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", size: 12, color: C.texto2 },
    margin: { l: 48, r: 12, t: movil ? 36 : 12, b: movil ? 84 : 36 },
    separators: ",.",
    hovermode: "x unified",
    hoverlabel: { bgcolor: C.tarjeta, bordercolor: C.borde, font: { color: C.texto, family: "Inter, sans-serif" } },
    // En móvil la leyenda va debajo del gráfico para no pisar los botones de rango
    legend: { orientation: "h", x: 0, y: movil ? -0.16 : 1.1, yanchor: movil ? "top" : "bottom",
              font: { size: 12, color: C.texto2 }, bgcolor: "rgba(0,0,0,0)" },
    xaxis: { type: "date", gridcolor: C.borde, linecolor: C.borde, zeroline: false, tickformatstops: FORMATO_FECHAS,
             tickfont: { color: C.texto3 } },
    yaxis: { title: { text: "$/barril", font: { size: 11, color: C.texto3 } }, gridcolor: C.borde,
             zeroline: false, tickfont: { color: C.texto3 } },
  };
}

// ---------- Gráfico principal ----------
function sombrasCiclos() {
  if (!estado.ciclosVisibles) return { shapes: [], annotations: [] };
  const shapes = estado.datos.ciclos.map((ci) => ({
    type: "rect", xref: "x", yref: "paper", x0: ci.inicio, x1: ci.fin, y0: 0, y1: 1,
    fillcolor: C.sombreado, line: { width: 0 }, layer: "below",
  }));
  const annotations = estado.datos.ciclos.map((ci, i) => ({
    xref: "x", yref: "paper", x: ci.inicio, y: i % 2 ? 0.93 : 0.99, xanchor: "left", yanchor: "top",
    text: ci.id === "2025" ? "2025-26" : ci.id, showarrow: false, font: { size: 10, color: C.texto3 },
  }));
  return { shapes, annotations };
}

function pintarPrincipal() {
  const { historico, prediccion } = estado.datos;
  const p = prediccion.horizontes.find((x) => x.h === estado.h);
  const t0 = prediccion.fecha_datos, t1 = p.fecha_objetivo_aprox, p0 = prediccion.precio_actual;
  const [li, ls] = p.intervalo_80;

  const trazas = [
    { x: historico.fechas, y: historico.precios, type: "scatter", mode: "lines", name: "Brent",
      line: { color: C.menta, width: 1.6 }, fill: "tozeroy",
      fillgradient: { type: "vertical", colorscale: [[0, "rgba(143,217,196,0)"], [1, "rgba(143,217,196,0.28)"]] },
      hovertemplate: "%{y:.2f} $<extra>Brent</extra>" },
    // Banda del 80 % en forma de cono: del precio actual al rango en t+h
    { x: [t0, t1, t1, t0], y: [p0, ls, li, p0], type: "scatter", mode: "lines", fill: "toself",
      fillcolor: C.ambarSuave, line: { width: 0 }, name: "Rango 80 %", hoverinfo: "skip" },
    { x: [t0, t1], y: [p0, p.precio_previsto], type: "scatter", mode: "lines+markers", name: "Previsión RNN",
      line: { color: C.ambar, width: 2, dash: "dash" },
      marker: { size: [0, 11], color: C.ambar, line: { width: 2, color: C.tarjeta } },
      hovertemplate: "%{y:.2f} $<extra>Previsión</extra>" },
    { x: [t1, t1], y: [li, ls], type: "scatter", mode: "markers", showlegend: false,
      marker: { symbol: "line-ew", size: 14, line: { width: 2, color: C.ambar } },
      hovertemplate: "%{y:.2f} $<extra>Rango 80 %</extra>" },
    { x: [t0, t1], y: [p0, p0], type: "scatter", mode: "lines", name: "Ingenuo",
      line: { color: C.texto2, width: 1.8, dash: "dot" }, hovertemplate: "%{y:.2f} $<extra>Ingenuo</extra>" },
  ];

  const fin = new Date(t1); fin.setDate(fin.getDate() + 20);
  const ini = new Date(t0); ini.setFullYear(ini.getFullYear() - 3);
  const layout = layoutBase();
  Object.assign(layout, sombrasCiclos());
  layout.xaxis.range = [ini.toISOString().slice(0, 10), fin.toISOString().slice(0, 10)];
  layout.xaxis.rangeselector = {
    buttons: [
      { count: 1, label: "1 año", step: "year", stepmode: "backward" },
      { count: 3, label: "3 años", step: "year", stepmode: "backward" },
      { count: 10, label: "10 años", step: "year", stepmode: "backward" },
      { step: "all", label: "Todo" },
    ],
    x: 1, xanchor: "right", y: 1.1, bgcolor: C.tarjeta2, activecolor: "rgba(143,217,196,0.25)",
    bordercolor: C.borde, borderwidth: 1, font: { size: 11, color: C.texto2 },
  };
  layout.yaxis.range = rangoY(layout.xaxis.range[0], layout.xaxis.range[1]);
  const div = document.getElementById("grafico-principal");
  Plotly.react(div, trazas, layout, CONFIG_PLOTLY);
  // Al cambiar el rango de fechas (botones o zoom), el eje Y se ajusta a lo visible
  if (!div.dataset.escucha) {
    div.dataset.escucha = "1";
    div.on("plotly_relayout", (ev) => {
      if (ev["yaxis.range[0]"] !== undefined || ev["yaxis.range"]) return;
      const r = div.layout.xaxis.range;
      Plotly.relayout(div, { "yaxis.range": rangoY(r[0], r[1]) });
    });
  }
}

// Rango del eje Y según las fechas visibles (histórico + previsión e intervalo)
function rangoY(x0, x1) {
  const { historico, prediccion } = estado.datos;
  const p = prediccion.horizontes.find((x) => x.h === estado.h);
  const a = String(x0).slice(0, 10), b = String(x1).slice(0, 10);
  let min = Infinity, max = -Infinity;
  historico.fechas.forEach((f, i) => {
    if (f >= a && f <= b) { min = Math.min(min, historico.precios[i]); max = Math.max(max, historico.precios[i]); }
  });
  if (p.fecha_objetivo_aprox <= b) { min = Math.min(min, p.intervalo_80[0]); max = Math.max(max, p.intervalo_80[1]); }
  if (!isFinite(min)) return undefined;
  const m = (max - min) * 0.06;
  return [Math.max(0, min - m), max + m];
}

// ---------- Medidor de cobertura ----------
function pintarGauge() {
  const c = filaComp(estado.datos.test.red, estado.h);
  document.getElementById("gauge-h").textContent = `horizonte ${CORTOS[estado.h]}`;
  const traza = {
    type: "indicator", mode: "gauge+number+delta", value: c["cobertura_80_%_ensemble"],
    number: { suffix: " %", valueformat: ".1f", font: { size: 30, color: C.texto, family: "Inter, sans-serif" } },
    delta: { reference: 80, valueformat: ".1f", suffix: " pp vs objetivo", font: { size: 12 },
             increasing: { color: C.menta, symbol: "▲ " }, decreasing: { color: C.rosa, symbol: "▼ " } },
    gauge: {
      shape: "angular",
      axis: { range: [0, 100], tickvals: [0, 25, 50, 80, 100], ticksuffix: " %", tickcolor: C.borde,
              tickfont: { size: 10, color: C.texto3 } },
      bar: { color: C.menta, thickness: 0.32 },
      bgcolor: C.tarjeta2, borderwidth: 0,
      threshold: { line: { color: C.ambar, width: 3 }, thickness: 0.9, value: 80 },
    },
  };
  Plotly.react("grafico-gauge", [traza], {
    paper_bgcolor: "rgba(0,0,0,0)", margin: { l: 28, r: 28, t: 18, b: 6 }, separators: ",.",
    font: { family: "Inter, system-ui, sans-serif", color: C.texto2 },
    annotations: [{ text: "objetivo 80 %", x: 0.86, y: 0.98, xref: "paper", yref: "paper", showarrow: false,
                    font: { size: 11, color: C.ambar } }],
  }, CONFIG_PLOTLY);
}

// ---------- Barras de error ----------
function pintarBarras() {
  const red = estado.datos.test.red;
  const hs = [5, 63, 252];
  const x = hs.map((h) => CORTOS[h]);
  const yRed = hs.map((h) => filaComp(red, h).RMSE_usd_ensemble);
  const yIng = hs.map((h) => filaComp("ingenuo", h).RMSE_usd_ensemble);
  const etiqueta = (v) => fmt(v, 1);
  const trazas = [
    { type: "bar", name: NOMBRES[red], x, y: yRed, marker: { color: C.ambar }, text: yRed.map(etiqueta),
      textposition: "outside", textfont: { color: C.texto2, size: 11 }, hovertemplate: "%{y:.2f} $<extra>RNN</extra>" },
    { type: "bar", name: "Ingenuo", x, y: yIng, marker: { color: C.texto3 }, text: yIng.map(etiqueta),
      textposition: "outside", textfont: { color: C.texto2, size: 11 }, hovertemplate: "%{y:.2f} $<extra>Ingenuo</extra>" },
  ];
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", separators: ",.",
    font: { family: "Inter, system-ui, sans-serif", size: 12, color: C.texto2 },
    margin: { l: 40, r: 8, t: 30, b: 28 }, barmode: "group", bargap: 0.35, bargroupgap: 0.12,
    legend: { orientation: "h", x: 0, y: 1.18, font: { size: 12, color: C.texto2 } },
    xaxis: { tickfont: { color: C.texto2 }, linecolor: C.borde },
    yaxis: { gridcolor: C.borde, zeroline: false, tickfont: { color: C.texto3 }, ticksuffix: " $",
             range: [0, Math.max(...yRed, ...yIng) * 1.22] },
    hoverlabel: { bgcolor: C.tarjeta, bordercolor: C.borde, font: { color: C.texto } },
  };
  Plotly.react("grafico-barras", trazas, layout, CONFIG_PLOTLY);
}

// ---------- Ciclos ----------
function pintarCiclos() {
  const cont = document.getElementById("lista-ciclos");
  cont.innerHTML = estado.datos.ciclos.map((ci) => `
    <article class="tarjeta ciclo">
      <p class="ciclo__periodo"><i data-lucide="calendar-range"></i>${fmtFecha(ci.inicio)} → ${fmtFecha(ci.fin)}</p>
      <h3>${ci.titulo}</h3>
      <p>${ci.texto}</p>
      ${ci.todo ? `<p class="ciclo__todo">Pendiente: ${ci.todo}</p>` : ""}
      ${ci.fuentes ? `<p class="ciclo__fuentes">Fuentes: ${ci.fuentes}</p>` : ""}
    </article>`).join("");
}

// ---------- Test ----------
function pintarTest() {
  const d = estado.datos.test.horizontes[String(estado.hTest)];
  const trazas = [
    { x: d.fechas.concat([...d.fechas].reverse()), y: d.red_ls.concat([...d.red_li].reverse()),
      type: "scatter", mode: "lines", fill: "toself", fillcolor: C.ambarSuave, line: { width: 0 },
      name: "Rango 80 % RNN", hoverinfo: "skip" },
    { x: d.fechas, y: d.real, type: "scatter", mode: "lines", name: "Brent real",
      line: { color: C.menta, width: 1.6 }, hovertemplate: "%{y:.2f} $<extra>Real</extra>" },
    { x: d.fechas, y: d.red, type: "scatter", mode: "lines", name: "RNN",
      line: { color: C.ambar, width: 1.8 }, hovertemplate: "%{y:.2f} $<extra>RNN</extra>" },
    { x: d.fechas, y: d.ingenuo, type: "scatter", mode: "lines", name: "Ingenuo",
      line: { color: C.texto2, width: 1.4, dash: "dot" }, hovertemplate: "%{y:.2f} $<extra>Ingenuo</extra>" },
  ];
  const layout = layoutBase();
  layout.xaxis.title = { text: "fecha objetivo", font: { size: 11, color: C.texto3 } };
  if (!estrecho()) layout.margin.b = 48;
  else { layout.margin.t = 12; layout.xaxis.title = null; }
  Plotly.react("grafico-test", trazas, layout, CONFIG_PLOTLY);
}

// ---------- Tabla ----------
function barraMejora(m) {
  // m en %, la barra se satura en ±50 %; el número siempre se muestra al lado
  const ancho = Math.min(Math.abs(m), 50);
  const clase = m >= 0 ? "pos" : "neg";
  return `<div class="mejora" title="Mejora vs ingenuo: ${fmtPct(m)}">
      <span class="mejora__pista"><span class="mejora__barra mejora__barra--${clase}" style="width:${ancho}%"></span></span>
      <span class="mejora__num ${m >= 0 ? "sube" : "baja"}">${fmtPct(m)}</span></div>`;
}

function pintarTabla() {
  const filas = estado.datos.comparativa;
  const red = estado.datos.test.red;
  const html = [];
  for (const h of [5, 63, 252]) {
    html.push(`<tr class="grupo"><td colspan="10">Horizonte ${h} días hábiles (${PLAZOS[h].split(" (")[0]})</td></tr>`);
    for (const f of filas.filter((x) => x.horizonte === h)) {
      const ing = f.modelo === "ingenuo";
      const dir = ing ? `${fmt(f.pct_dias_subida, 1)} % (subidas)` : `${fmt(f["acierto_direccion_%_ensemble"], 1)} %`;
      const tiempo = ing ? "—" : fmt(f.segundos_ajuste + f.segundos_final, 0);
      html.push(`<tr class="${f.modelo === red ? "elegido" : ""}">
        <td>${NOMBRES[f.modelo] || f.modelo}${f.modelo === red ? '<span class="chip">en uso</span>' : ""}</td>
        <td class="num">${fmt(f.MAE_usd_ensemble)}</td>
        <td class="num">${fmt(f.RMSE_usd_ensemble)}</td>
        <td>${ing ? '<span class="texto-sec">referencia</span>' : barraMejora(100 * f.mejora_vs_ingenuo_ensemble)}</td>
        <td class="centro">${ing ? "—" : `<span class="pastilla">p = ${fmt(f.DM_p_valor, 2)}</span>`}</td>
        <td class="num">${fmt(f.sesgo_usd_ensemble)}</td>
        <td class="num">${dir}</td>
        <td class="num">${fmt(f["cobertura_80_%_ensemble"], 1)} %</td>
        <td class="num">${f.n_parametros.toLocaleString("es-ES")}</td>
        <td class="num">${tiempo}</td></tr>`);
    }
  }
  document.querySelector("#tabla-comp tbody").innerHTML = html.join("");
}

// ---------- Navegación: sección activa y título de la barra superior ----------
function navegacionActiva() {
  const enlaces = [...document.querySelectorAll("#nav a")];
  const titulo = document.getElementById("titulo-seccion");
  const secciones = [...document.querySelectorAll("section[data-titulo]")];
  const activar = (id) => {
    enlaces.forEach((a) => {
      const activo = a.dataset.seccion === id;
      a.classList.toggle("activo", activo);
      if (activo) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current");
    });
    const s = secciones.find((x) => x.id === id);
    if (s) titulo.textContent = s.dataset.titulo;
  };
  // La sección activa es la última cuyo inicio ha pasado por debajo de la barra superior
  const alDesplazar = () => {
    const limite = window.innerHeight * 0.3;
    let actual = secciones[0].id;
    for (const s of secciones) if (s.getBoundingClientRect().top <= limite) actual = s.id;
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) actual = secciones[secciones.length - 1].id;
    activar(actual);
  };
  window.addEventListener("scroll", alDesplazar, { passive: true });
  alDesplazar();
}

// ---------- Controles ----------
function segmentado(id, alCambiar) {
  const grupo = document.getElementById(id);
  grupo.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-h]");
    if (!b) return;
    grupo.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    alCambiar(Number(b.dataset.h));
  });
}

function conectarControles() {
  segmentado("selector-h", (h) => { estado.h = h; actualizarHorizonte(); });
  segmentado("selector-test", (h) => { estado.hTest = h; pintarTest(); });
  document.getElementById("chk-ciclos").addEventListener("change", (e) => {
    estado.ciclosVisibles = e.target.checked;
    Plotly.relayout("grafico-principal", sombrasCiclos());
  });
}

iniciar();
