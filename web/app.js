/* =============================================================
   Previsión del Brent: lógica de la web.
   La web no calcula nada: solo lee los JSON de data/ (generados en Python).
   ============================================================= */
"use strict";

const estado = { h: 5, hTest: 5, ciclosVisibles: true, datos: null };

// ---------- Paleta (terminal: negro, ámbar único, verde/rojo solo para cifras con signo) ----------
const C = {
  texto: "#E8E8E8", texto2: "#8A8A8A", texto3: "#5E5E5E", linea: "#222222", rejilla: "#161616",
  historico: "#CFCFCF", ambar: "#FFB000", ambar18: "rgba(255,176,0,0.18)", sombreado: "rgba(255,255,255,0.045)",
};
const FUENTE = "Inter, system-ui, sans-serif";

// ---------- Formato ----------
const fmt = (x, d = 2) => x == null || Number.isNaN(x) ? "—"
  : x.toLocaleString("es-ES", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtPct = (x, d = 1) => x == null ? "—" : (x > 0 ? "+" : "") + fmt(x, d) + " %";
const claseSigno = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");
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
const estrecho = () => window.innerWidth < 600;

// ---------- Carga ----------
async function cargar(nombre) {
  const r = await fetch(`data/${nombre}.json`, { cache: "no-cache" });
  if (!r.ok) throw new Error(`No se pudo cargar ${nombre}.json`);
  return r.json();
}

async function iniciar() {
  try {
    const [historico, prediccion, comparativa, test, ciclos, meta] = await Promise.all(
      ["historico", "prediccion", "comparativa", "test", "ciclos", "meta"].map(cargar));
    estado.datos = { historico, prediccion, comparativa, test, ciclos, meta };
  } catch (e) {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<div class="panel">No se han podido cargar los datos (${e.message}).
       Si abres el fichero directamente, sírvelo con <code>python -m http.server</code>.</div>`);
    return;
  }
  pintarCabecera();
  pintarCiclos();
  pintarTabla();
  conectarControles();
  actualizarHorizonte();
  pintarTest();
  navegacionActiva();
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
  tVar.className = "kpi__cifra " + claseSigno(p.variacion_pct);
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
  pintarMejora();
  pintarCobertura();
}

// ---------- Estilo común de Plotly ----------
const FORMATO_FECHAS = [
  { dtickrange: [null, "M1"], value: "%d-%m-%Y" },
  { dtickrange: ["M1", "M12"], value: "%m-%Y" },
  { dtickrange: ["M12", null], value: "%Y" },
];

function layoutBase() {
  const movil = estrecho();
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: FUENTE, size: 11, color: C.texto2 },
    margin: { l: 44, r: 8, t: movil ? 34 : 10, b: movil ? 82 : 34 },
    separators: ",.",
    hovermode: "x unified",
    hoverlabel: { bgcolor: "#000", bordercolor: C.linea, font: { color: C.texto, family: FUENTE, size: 12 } },
    // En móvil la leyenda va debajo del gráfico para no pisar los botones de rango
    legend: { orientation: "h", x: 0, y: movil ? -0.16 : 1.08, yanchor: movil ? "top" : "bottom",
              font: { size: 11, color: C.texto2 }, bgcolor: "rgba(0,0,0,0)" },
    xaxis: { type: "date", gridcolor: C.rejilla, linecolor: C.linea, zeroline: false, tickformatstops: FORMATO_FECHAS,
             tickfont: { color: C.texto2, size: 11 } },
    yaxis: { title: { text: "$/barril", font: { size: 11, color: C.texto3 } }, gridcolor: C.rejilla,
             zeroline: false, tickfont: { color: C.texto2, size: 11 } },
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
      line: { color: C.historico, width: 1.2 }, hovertemplate: "%{y:.2f} $<extra>Brent</extra>" },
    // Banda del 80 % en forma de cono: del precio actual al rango en t+h
    { x: [t0, t1, t1, t0], y: [p0, ls, li, p0], type: "scatter", mode: "lines", fill: "toself",
      fillcolor: C.ambar18, line: { width: 0 }, name: "Rango 80 %", hoverinfo: "skip" },
    { x: [t0, t1], y: [p0, p.precio_previsto], type: "scatter", mode: "lines+markers", name: "Previsión RNN",
      line: { color: C.ambar, width: 1.6, dash: "dash" },
      marker: { size: [0, 9], color: C.ambar, symbol: "square" },
      hovertemplate: "%{y:.2f} $<extra>Previsión</extra>" },
    { x: [t1, t1], y: [li, ls], type: "scatter", mode: "markers", showlegend: false,
      marker: { symbol: "line-ew", size: 14, line: { width: 1.6, color: C.ambar } },
      hovertemplate: "%{y:.2f} $<extra>Rango 80 %</extra>" },
    { x: [t0, t1], y: [p0, p0], type: "scatter", mode: "lines", name: "Ingenuo",
      line: { color: C.texto2, width: 1.2, dash: "dash" }, hovertemplate: "%{y:.2f} $<extra>Ingenuo</extra>" },
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
    x: 1, xanchor: "right", y: 1.08, yanchor: "bottom", bgcolor: "#000", activecolor: "rgba(255,176,0,0.30)",
    bordercolor: C.linea, borderwidth: 1, font: { size: 11, color: C.texto },
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

// ---------- Panel lateral: mejora vs ingenuo ----------
function pintarMejora() {
  const hs = [5, 63, 252];
  const red = estado.datos.test.red;
  document.querySelectorAll("#tabla-mejora th[data-h]").forEach((th) =>
    th.classList.toggle("actual", Number(th.dataset.h) === estado.h));
  document.querySelector("#tabla-mejora tbody").innerHTML = ["rnn", "lstm", "gru"].map((m) => `
    <tr><td>${NOMBRES[m]}${m === red ? '<span class="marca-uso">en uso</span>' : ""}</td>
      ${hs.map((h) => {
        const v = 100 * filaComp(m, h).mejora_vs_ingenuo_ensemble;
        return `<td class="num ${claseSigno(v)} ${h === estado.h ? "actual" : ""}">${fmtPct(v)}</td>`;
      }).join("")}</tr>`).join("");
  document.getElementById("dm-rnn").textContent =
    hs.map((h) => `${CORTOS[h]} ${fmt(filaComp(red, h).DM_p_valor, 2)}`).join(" · ");
}

// ---------- Panel lateral: cobertura del intervalo ----------
function pintarCobertura() {
  const red = estado.datos.test.red;
  document.getElementById("cob-h").textContent = CORTOS[estado.h];
  const fila = (modelo, clase) => {
    const v = filaComp(modelo, estado.h)["cobertura_80_%_ensemble"];
    return `<div class="cob__fila">
        <span class="cob__nombre">${NOMBRES[modelo]}</span>
        <span class="cob__pista" role="img" aria-label="${NOMBRES[modelo]}: ${fmt(v, 1)} % de cobertura">
          <span class="cob__barra ${clase}" style="width:${Math.min(v, 100)}%"></span>
          <span class="cob__marca" style="left:calc(80% - 1px)"></span></span>
        <span class="cob__valor">${fmt(v, 1)} %</span></div>`;
  };
  document.getElementById("cobertura").innerHTML = `<div class="cob">
      ${fila(red, "cob__barra--red")}${fila("ingenuo", "")}
      <div class="cob__escala"><span></span><span><i style="left:0">0</i><i style="left:50%">50</i>
        <i style="left:80%;color:#FFB000">80</i><i style="left:100%">100</i></span><span></span></div></div>`;
}

// ---------- Ciclos ----------
function pintarCiclos() {
  const cont = document.getElementById("lista-ciclos");
  cont.innerHTML = estado.datos.ciclos.map((ci) => `
    <article class="panel ciclo">
      <p class="etq">${fmtFecha(ci.inicio)} → ${fmtFecha(ci.fin)}</p>
      <h3 class="titulo">${ci.titulo}</h3>
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
      type: "scatter", mode: "lines", fill: "toself", fillcolor: C.ambar18, line: { width: 0 },
      name: "Rango 80 % RNN", hoverinfo: "skip" },
    { x: d.fechas, y: d.real, type: "scatter", mode: "lines", name: "Brent real",
      line: { color: C.historico, width: 1.2 }, hovertemplate: "%{y:.2f} $<extra>Real</extra>" },
    { x: d.fechas, y: d.red, type: "scatter", mode: "lines", name: "RNN",
      line: { color: C.ambar, width: 1.4 }, hovertemplate: "%{y:.2f} $<extra>RNN</extra>" },
    { x: d.fechas, y: d.ingenuo, type: "scatter", mode: "lines", name: "Ingenuo",
      line: { color: C.texto2, width: 1.1, dash: "dash" }, hovertemplate: "%{y:.2f} $<extra>Ingenuo</extra>" },
  ];
  const layout = layoutBase();
  if (!estrecho()) {
    layout.xaxis.title = { text: "fecha objetivo", font: { size: 11, color: C.texto3 } };
    layout.margin.b = 46;
  } else {
    layout.margin.t = 10;
  }
  Plotly.react("grafico-test", trazas, layout, CONFIG_PLOTLY);
}

// ---------- Tabla comparativa ----------
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
      const mejora = 100 * f.mejora_vs_ingenuo_ensemble;
      html.push(`<tr class="${f.modelo === red ? "elegido" : ""}">
        <td>${NOMBRES[f.modelo] || f.modelo}${f.modelo === red ? '<span class="marca-uso">en uso</span>' : ""}</td>
        <td class="num">${fmt(f.MAE_usd_ensemble)}</td>
        <td class="num">${fmt(f.RMSE_usd_ensemble)}</td>
        <td class="num ${ing ? "" : claseSigno(mejora)}">${ing ? "referencia" : fmtPct(mejora)}</td>
        <td class="num">${ing ? "—" : `<span class="etiqueta-dm">p = ${fmt(f.DM_p_valor, 2)}</span>`}</td>
        <td class="num">${fmt(f.sesgo_usd_ensemble)}</td>
        <td class="num">${dir}</td>
        <td class="num">${fmt(f["cobertura_80_%_ensemble"], 1)} %</td>
        <td class="num">${f.n_parametros.toLocaleString("es-ES")}</td>
        <td class="num">${tiempo}</td></tr>`);
    }
  }
  document.querySelector("#tabla-comp tbody").innerHTML = html.join("");
}

// ---------- Navegación: sección activa subrayada ----------
function navegacionActiva() {
  const enlaces = [...document.querySelectorAll("#nav a")];
  const secciones = enlaces.map((a) => document.getElementById(a.dataset.seccion)).filter(Boolean);
  const activar = (id) => enlaces.forEach((a) => {
    const activo = a.dataset.seccion === id;
    a.classList.toggle("activo", activo);
    if (activo) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current");
  });
  // La sección activa es la última cuyo inicio ha pasado por debajo de la barra superior
  const alDesplazar = () => {
    const limite = Math.min(window.innerHeight * 0.3, 260);
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
