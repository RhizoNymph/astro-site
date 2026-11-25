import { useState, useRef, useCallback, useEffect, useMemo } from "react";

/* ─── PRNG ─── */
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ─── DAG Generation ─── */
function generateDAG(nodeCount, edgeDensity, seed, rootSide) {
  const rng = mulberry32(seed);
  const nodes = [];
  const edges = [];
  const layers = Math.max(3, Math.ceil(Math.sqrt(nodeCount) * 1.4));
  const nodesPerLayer = [];
  let remaining = nodeCount;

  for (let i = 0; i < layers; i++) {
    const avg = remaining / (layers - i);
    const count = i === layers - 1 ? remaining : Math.max(1, Math.round(avg + (rng() - 0.5) * avg * 0.6));
    nodesPerLayer.push(Math.min(count, remaining));
    remaining -= nodesPerLayer[i];
    if (remaining <= 0) break;
  }

  const actualLayers = nodesPerLayer.filter(c => c > 0);
  let id = 0;
  const W = 880, H = 520, PAD = 60;

  actualLayers.forEach((count, li) => {
    for (let i = 0; i < count; i++) {
      const layerT = actualLayers.length === 1 ? 0.5 : li / (actualLayers.length - 1);
      const slotT = count === 1 ? 0.5 : i / (count - 1);
      let x, y;

      if (rootSide === "left" || rootSide === "right") {
        const rawX = PAD + layerT * (W - PAD * 2);
        x = rootSide === "right" ? W - rawX + PAD : rawX;
        y = PAD + slotT * (H - PAD * 2) + (rng() - 0.5) * 28;
        x += (rng() - 0.5) * 35;
      } else {
        const rawY = PAD + layerT * (H - PAD * 2);
        y = rootSide === "bottom" ? H - rawY + PAD : rawY;
        x = PAD + slotT * (W - PAD * 2) + (rng() - 0.5) * 28;
        y += (rng() - 0.5) * 25;
      }

      nodes.push({ id: id++, layer: li, x, y, label: `n${id - 1}` });
    }
  });

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      if (nodes[i].layer < nodes[j].layer && rng() < edgeDensity) {
        edges.push({ from: nodes[i].id, to: nodes[j].id });
      }
    }
  }

  const hasParent = new Set();
  edges.forEach(e => hasParent.add(e.to));
  nodes.forEach(n => {
    if (n.layer > 0 && !hasParent.has(n.id)) {
      const candidates = nodes.filter(c => c.layer < n.layer);
      if (candidates.length > 0) {
        const parent = candidates[Math.floor(rng() * candidates.length)];
        edges.push({ from: parent.id, to: n.id });
      }
    }
  });

  return { nodes, edges };
}

/* ─── Topological Sort ─── */
function topoSort(nodes, edges) {
  const adj = {}, inDeg = {};
  nodes.forEach(n => { adj[n.id] = []; inDeg[n.id] = 0; });
  edges.forEach(e => { adj[e.from].push(e.to); inDeg[e.to]++; });
  const queue = nodes.filter(n => inDeg[n.id] === 0).map(n => n.id);
  const order = [];
  while (queue.length) {
    queue.sort((a, b) => a - b);
    const u = queue.shift();
    order.push(u);
    for (const v of adj[u]) {
      inDeg[v]--;
      if (inDeg[v] === 0) queue.push(v);
    }
  }
  return order;
}

/* ─── Critical Path ─── */
function computeCriticalPath(nodes, edges) {
  const adj = {}, inDeg = {};
  nodes.forEach(n => { adj[n.id] = []; inDeg[n.id] = 0; });
  edges.forEach(e => { adj[e.from].push(e.to); inDeg[e.to]++; });
  const dist = {}, prev = {};
  nodes.forEach(n => { dist[n.id] = 0; prev[n.id] = null; });
  const queue = nodes.filter(n => inDeg[n.id] === 0).map(n => n.id);
  while (queue.length) {
    const u = queue.shift();
    for (const v of adj[u]) {
      if (dist[u] + 1 > dist[v]) { dist[v] = dist[u] + 1; prev[v] = u; }
      inDeg[v]--;
      if (inDeg[v] === 0) queue.push(v);
    }
  }
  let maxD = 0, end = null;
  nodes.forEach(n => { if (dist[n.id] > maxD) { maxD = dist[n.id]; end = n.id; } });
  const path = new Set();
  let c = end;
  while (c !== null) { path.add(c); c = prev[c]; }
  return path;
}

function bezierPath(x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  if (Math.abs(dx) > Math.abs(dy)) {
    return `M ${x1} ${y1} C ${x1 + dx * 0.4} ${y1}, ${x1 + dx * 0.6} ${y2}, ${x2} ${y2}`;
  } else {
    return `M ${x1} ${y1} C ${x1} ${y1 + dy * 0.4}, ${x2} ${y1 + dy * 0.6}, ${x2} ${y2}`;
  }
}

const NODE_R = 20;

/* ─── Presets ─── */
const PRESETS = [
  { name: "Phosphor", bg: "#0a0e14", node: "#0f1922", stroke: "#00e5a0", accent: "#00e5a0", edge: "#1a3a2e", text: "#c0ffe0", critical: "#ff6b6b" },
  { name: "Amber", bg: "#0d0a06", node: "#1a1408", stroke: "#ffb300", accent: "#ffb300", edge: "#3a2e10", text: "#ffe0a0", critical: "#ff4444" },
  { name: "Infrared", bg: "#0e0609", node: "#1c0a12", stroke: "#ff3366", accent: "#ff3366", edge: "#3a1020", text: "#ffc0d0", critical: "#44ddff" },
  { name: "Coolant", bg: "#060a10", node: "#0a1220", stroke: "#3399ff", accent: "#3399ff", edge: "#102040", text: "#b0d0ff", critical: "#ff8844" },
  { name: "Orchid", bg: "#0c060e", node: "#180c20", stroke: "#cc66ff", accent: "#cc66ff", edge: "#2a1040", text: "#e0c0ff", critical: "#66ffaa" },
  { name: "Bone", bg: "#f5f0eb", node: "#ebe5dd", stroke: "#2a2420", accent: "#8b4513", edge: "#d0c8bb", text: "#3a3430", critical: "#c04020" },
];

const COLOR_KEYS = [
  { key: "bg", label: "Background" },
  { key: "node", label: "Node Fill" },
  { key: "stroke", label: "Stroke" },
  { key: "accent", label: "Accent" },
  { key: "edge", label: "Edge" },
  { key: "text", label: "Labels" },
  { key: "critical", label: "Critical" },
];

function isLight(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return (r * 299 + g * 587 + b * 114) / 1000 > 128;
}

function Ctrl({ label, value, children }) {
  return (
    <div className="ctrl">
      <span className="lbl">{label}</span>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {children}
        <span className="val">{value}</span>
      </div>
    </div>
  );
}

/* ─── Main Component ─── */
export default function DAGGenerator() {
  const [nodeCount, setNodeCount] = useState(14);
  const [density, setDensity] = useState(0.3);
  const [seed, setSeed] = useState(42);
  const [rootSide, setRootSide] = useState("left");
  const [palette, setPalette] = useState({ ...PRESETS[0] });
  const [showCritical, setShowCritical] = useState(true);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [paletteOpen, setPaletteOpen] = useState(false);

  /* ─── Animation State ─── */
  const [animating, setAnimating] = useState(false);
  const [walkSpeed, setWalkSpeed] = useState(350);
  const [animVisited, setAnimVisited] = useState(new Set());     // visited node ids
  const [animWalked, setAnimWalked] = useState(new Set());       // previously walked edge indices (accent)
  const [animFrontier, setAnimFrontier] = useState(new Set());   // current frontier edge indices (critical)
  const [animActiveNode, setAnimActiveNode] = useState(null);    // currently being visited node
  const [fadeOpacity, setFadeOpacity] = useState(1);
  const timerRef = useRef(null);
  const cancelledRef = useRef(false);

  const dag = useMemo(() => generateDAG(nodeCount, density, seed, rootSide), [nodeCount, density, seed, rootSide]);
  const critPath = useMemo(() => computeCriticalPath(dag.nodes, dag.edges), [dag]);

  const nodeMap = useMemo(() => {
    const m = {}; dag.nodes.forEach(n => m[n.id] = n); return m;
  }, [dag]);

  const inDeg = useMemo(() => {
    const d = {}; dag.nodes.forEach(n => d[n.id] = 0); dag.edges.forEach(e => d[e.to]++); return d;
  }, [dag]);
  const outDeg = useMemo(() => {
    const d = {}; dag.nodes.forEach(n => d[n.id] = 0); dag.edges.forEach(e => d[e.from]++); return d;
  }, [dag]);

  /* ─── Animation Engine ─── */
  const clearTimers = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  const stopAnim = useCallback(() => {
    cancelledRef.current = true;
    clearTimers();
    setAnimating(false);
    setAnimVisited(new Set());
    setAnimWalked(new Set());
    setAnimFrontier(new Set());
    setAnimActiveNode(null);
    setFadeOpacity(1);
  }, [clearTimers]);

  const scheduleStep = useCallback((fn, ms) => {
    return new Promise(resolve => {
      timerRef.current = setTimeout(() => {
        if (!cancelledRef.current) { fn(); resolve(true); }
        else resolve(false);
      }, ms);
    });
  }, []);

  const runCycle = useCallback(async (startSeed) => {
    cancelledRef.current = false;
    let currentSeed = startSeed;

    while (!cancelledRef.current) {
      const dagCycle = generateDAG(nodeCount, density, currentSeed, rootSide);
      const order = topoSort(dagCycle.nodes, dagCycle.edges);

      setSeed(currentSeed);
      setAnimVisited(new Set());
      setAnimWalked(new Set());
      setAnimFrontier(new Set());
      setAnimActiveNode(null);
      setFadeOpacity(1);

      // Pause before walk starts
      const ok0 = await scheduleStep(() => {}, 600);
      if (!ok0) break;

      const visited = new Set();
      const walked = new Set();
      let broken = false;

      for (let step = 0; step < order.length; step++) {
        if (cancelledRef.current) { broken = true; break; }
        const nid = order[step];

        // Find incoming edges to this node from already-visited nodes = frontier
        const frontier = new Set();
        dagCycle.edges.forEach((e, idx) => {
          if (e.to === nid && visited.has(e.from)) frontier.add(idx);
        });

        // Move previous frontier edges into walked
        // (walked already contains everything from before)
        visited.add(nid);

        const snapVisited = new Set(visited);
        const snapWalked = new Set(walked);
        const snapFrontier = new Set(frontier);

        const ok = await scheduleStep(() => {
          setAnimActiveNode(nid);
          setAnimVisited(snapVisited);
          setAnimWalked(snapWalked);
          setAnimFrontier(snapFrontier);
        }, walkSpeed);
        if (!ok) { broken = true; break; }

        // After showing frontier, merge frontier into walked for next step
        frontier.forEach(idx => walked.add(idx));
      }
      if (broken) break;

      // Clear frontier on final hold, all edges are now walked
      const okSettle = await scheduleStep(() => {
        setAnimFrontier(new Set());
        setAnimWalked(new Set(walked));
        setAnimActiveNode(null);
      }, walkSpeed);
      if (!okSettle) break;

      // Hold fully walked
      const okHold = await scheduleStep(() => {}, 1400);
      if (!okHold) break;

      // Fade out
      const okFade = await scheduleStep(() => setFadeOpacity(0), 50);
      if (!okFade) break;

      // Wait for CSS fade
      const okBlank = await scheduleStep(() => {
        setAnimVisited(new Set());
        setAnimWalked(new Set());
        setAnimFrontier(new Set());
        setAnimActiveNode(null);
      }, 900);
      if (!okBlank) break;

      // New seed, fade back in
      currentSeed = Math.floor(Math.random() * 100000);
      const okNext = await scheduleStep(() => setFadeOpacity(1), 300);
      if (!okNext) break;
    }
  }, [nodeCount, density, rootSide, walkSpeed, scheduleStep]);

  const startAnim = useCallback(() => {
    stopAnim();
    cancelledRef.current = false;
    setAnimating(true);
    setHoveredNode(null);
    // Use setTimeout to let state settle before starting async loop
    setTimeout(() => runCycle(seed), 50);
  }, [seed, runCycle, stopAnim]);

  useEffect(() => {
    return () => { cancelledRef.current = true; clearTimers(); };
  }, [clearTimers]);

  const applyPreset = (p) => setPalette({ ...p });
  const updateColor = (key, val) => setPalette(prev => ({ ...prev, [key]: val }));

  /* Hover connections */
  const connSet = new Set();
  if (hoveredNode !== null && !animating) {
    dag.edges.forEach(e => {
      if (e.from === hoveredNode) connSet.add(e.to);
      if (e.to === hoveredNode) connSet.add(e.from);
    });
    connSet.add(hoveredNode);
  }

  return (
    <div style={{
      minHeight: "100vh", background: palette.bg,
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      color: palette.text, padding: "20px",
      transition: "background 0.5s, color 0.5s",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');
        * { box-sizing: border-box; margin: 0; }
        .panel {
          padding: 14px 18px; border-radius: 6px; margin-bottom: 12px;
          background: ${palette.node}; border: 1px solid ${palette.edge};
          transition: background 0.4s, border-color 0.4s;
        }
        .row { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; }
        .ctrl { display: flex; flex-direction: column; gap: 3px; }
        .lbl {
          font-size: 8px; text-transform: uppercase; letter-spacing: 2px;
          color: ${palette.accent}; opacity: 0.6; font-weight: 500;
        }
        .val { font-size: 12px; font-weight: 300; }
        input[type="range"] {
          -webkit-appearance: none; width: 105px; height: 2px;
          background: ${palette.edge}; border-radius: 1px; outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none; width: 12px; height: 12px;
          border-radius: 50%; background: ${palette.accent}; cursor: pointer;
        }
        .btn {
          padding: 6px 13px; border-radius: 3px; border: 1px solid ${palette.accent};
          background: transparent; color: ${palette.accent}; cursor: pointer;
          font-family: inherit; font-size: 10px; letter-spacing: 1px;
          text-transform: uppercase; transition: all 0.2s; font-weight: 500;
          white-space: nowrap;
        }
        .btn:hover { background: ${palette.accent}; color: ${palette.bg}; }
        .btn.on { background: ${palette.accent}18; }
        .btn.stop { border-color: ${palette.critical}; color: ${palette.critical}; }
        .btn.stop:hover { background: ${palette.critical}; color: ${palette.bg}; }
        .sbtn {
          width: 28px; height: 28px; border-radius: 3px; cursor: pointer;
          border: 1px solid ${palette.edge}; background: transparent;
          color: ${palette.text}; font-size: 13px; display: flex;
          align-items: center; justify-content: center; transition: all 0.15s;
          font-family: inherit;
        }
        .sbtn:hover, .sbtn.sel { border-color: ${palette.accent}; color: ${palette.accent}; background: ${palette.accent}10; }
        .color-row { display: flex; align-items: center; gap: 8px; padding: 3px 0; }
        .color-label { font-size: 10px; width: 76px; opacity: 0.6; font-weight: 300; }
        .cswatch {
          width: 24px; height: 24px; border-radius: 3px; cursor: pointer;
          border: 1px solid ${palette.edge}; position: relative; overflow: hidden;
          flex-shrink: 0;
        }
        .cswatch input {
          position: absolute; inset: -6px; width: 38px; height: 38px;
          cursor: pointer; opacity: 0;
        }
        .chex {
          font-size: 10px; opacity: 0.5; width: 64px;
          background: transparent; border: 1px solid ${palette.edge};
          border-radius: 3px; padding: 2px 5px; color: ${palette.text};
          font-family: inherit; outline: none;
        }
        .chex:focus { border-color: ${palette.accent}; opacity: 0.9; }
        .pdot {
          width: 16px; height: 16px; border-radius: 50%; cursor: pointer;
          border: 2px solid transparent; transition: all 0.15s;
        }
        .pdot:hover { transform: scale(1.3); border-color: ${palette.text}; }
        .svg-wrap {
          border-radius: 6px; border: 1px solid ${palette.edge};
          background: ${palette.bg}; overflow: hidden;
          transition: background 0.4s, border-color 0.4s, opacity 0.8s ease-in-out;
        }
        .stats {
          display: flex; flex-wrap: wrap; gap: 16px; padding: 8px 18px;
          font-size: 10px; opacity: 0.45; letter-spacing: 1px; font-weight: 300;
        }
        .sv { color: ${palette.accent}; font-weight: 500; }
      `}</style>

      <div style={{ maxWidth: 1000, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 12 }}>
          <h1 style={{
            fontSize: 14, fontWeight: 700, letterSpacing: 3,
            textTransform: "uppercase", color: palette.accent,
          }}>DAG Generator</h1>
          <span style={{ fontSize: 9, opacity: 0.3 }}>seed:{seed}</span>
          {animating && <span style={{ fontSize: 9, color: palette.critical, opacity: 0.8 }}>● walking</span>}
        </div>

        {/* Controls */}
        <div className="panel">
          <div className="row">
            <Ctrl label="Nodes" value={nodeCount}>
              <input type="range" min={4} max={30} value={nodeCount} onChange={e => setNodeCount(+e.target.value)} />
            </Ctrl>
            <Ctrl label="Density" value={`${Math.round(density * 100)}%`}>
              <input type="range" min={10} max={80} value={Math.round(density * 100)} onChange={e => setDensity(+e.target.value / 100)} />
            </Ctrl>
            <Ctrl label="Speed" value={`${walkSpeed}ms`}>
              <input type="range" min={60} max={900} step={10} value={walkSpeed} onChange={e => setWalkSpeed(+e.target.value)} />
            </Ctrl>
            <div className="ctrl">
              <span className="lbl">Root</span>
              <div style={{ display: "flex", gap: 3 }}>
                {[["left","←"],["top","↑"],["right","→"],["bottom","↓"]].map(([s,arr]) => (
                  <button key={s} className={`sbtn ${rootSide === s ? "sel" : ""}`} onClick={() => setRootSide(s)}>{arr}</button>
                ))}
              </div>
            </div>
            <button className={`btn ${showCritical ? "on" : ""}`} onClick={() => setShowCritical(!showCritical)}>
              {showCritical ? "◆" : "◇"} Crit
            </button>
            <button className={`btn ${paletteOpen ? "on" : ""}`} onClick={() => setPaletteOpen(!paletteOpen)}>◐ Palette</button>
            {!animating
              ? <button className="btn" onClick={startAnim}>▶ Walk</button>
              : <button className="btn stop" onClick={stopAnim}>■ Stop</button>
            }
            <button className="btn" onClick={() => { stopAnim(); setSeed(Math.floor(Math.random() * 100000)); }}>↻ New</button>
          </div>
        </div>

        {/* Palette Editor */}
        {paletteOpen && (
          <div className="panel" style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
            <div style={{ minWidth: 220 }}>
              <div className="lbl" style={{ marginBottom: 8 }}>Presets</div>
              <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
                {PRESETS.map((p, i) => (
                  <div key={i} className="pdot" style={{ background: p.stroke }} title={p.name} onClick={() => applyPreset(p)} />
                ))}
              </div>
              {COLOR_KEYS.map(({ key, label }) => (
                <div className="color-row" key={key}>
                  <span className="color-label">{label}</span>
                  <div className="cswatch" style={{ background: palette[key] }}>
                    <input type="color" value={palette[key]} onChange={e => updateColor(key, e.target.value)} />
                  </div>
                  <input className="chex" value={palette[key]}
                    onChange={e => { if (/^#[0-9a-fA-F]{6}$/.test(e.target.value)) updateColor(key, e.target.value); }}
                  />
                </div>
              ))}
            </div>
            <div style={{ flex: 1, minWidth: 180 }}>
              <div className="lbl" style={{ marginBottom: 8 }}>Preview</div>
              <div style={{
                width: "100%", height: 100, borderRadius: 4,
                background: `linear-gradient(135deg, ${palette.bg}, ${palette.node} 50%, ${palette.edge})`,
                display: "flex", alignItems: "center", justifyContent: "center", gap: 18,
                border: `1px solid ${palette.edge}`,
              }}>
                <svg width="70" height="60" viewBox="0 0 70 60">
                  <line x1="14" y1="30" x2="56" y2="12" stroke={palette.edge} strokeWidth="1.5" />
                  <line x1="14" y1="30" x2="56" y2="48" stroke={palette.critical} strokeWidth="2" />
                  <circle cx="14" cy="30" r="9" fill={palette.node} stroke={palette.stroke} strokeWidth="1.5" />
                  <circle cx="56" cy="12" r="9" fill={palette.node} stroke={palette.edge} strokeWidth="1" />
                  <circle cx="56" cy="48" r="9" fill={palette.node} stroke={palette.critical} strokeWidth="1.5" />
                </svg>
                <div style={{ fontSize: 10, lineHeight: 2, fontWeight: 300 }}>
                  <div style={{ color: palette.accent }}>accent</div>
                  <div style={{ color: palette.text, opacity: 0.7 }}>label</div>
                  <div style={{ color: palette.critical }}>critical</div>
                </div>
              </div>
              <div style={{ marginTop: 10, fontSize: 9, opacity: 0.4, lineHeight: 1.6 }}>
                Tip: for a background image, try low contrast between bg/node/edge with a vivid accent.
              </div>
            </div>
          </div>
        )}

        {/* SVG Canvas */}
        <div className="svg-wrap" style={{ opacity: fadeOpacity }}>
          <svg viewBox="0 0 960 600" width="100%" style={{ display: "block" }}>
            <defs>
              <filter id="g1"><feGaussianBlur stdDeviation="3" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
              <filter id="g2"><feGaussianBlur stdDeviation="6" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
              <filter id="g3"><feGaussianBlur stdDeviation="12" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
              <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill={palette.accent} opacity="0.5" />
              </marker>
              <marker id="arrC" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill={palette.critical} opacity="0.8" />
              </marker>
              <marker id="arrL" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill={palette.accent} />
              </marker>
            </defs>

            {/* Grid dots */}
            {Array.from({ length: 24 }, (_, i) =>
              Array.from({ length: 15 }, (_, j) => (
                <circle key={`d${i}-${j}`} cx={i * 40 + 20} cy={j * 40 + 20}
                  r="0.5" fill={palette.edge} opacity="0.3" />
              ))
            )}

            {/* Edges */}
            {dag.edges.map((e, i) => {
              const from = nodeMap[e.from], to = nodeMap[e.to];
              const dx = to.x - from.x, dy = to.y - from.y;
              const d = Math.sqrt(dx * dx + dy * dy) || 1;
              const ox = (dx / d) * NODE_R, oy = (dy / d) * NODE_R;
              const isCrit = showCritical && !animating && critPath.has(e.from) && critPath.has(e.to);

              let opacity, stroke, width, marker, filt;
              if (animating) {
                const isFrontier = animFrontier.has(i);
                const isWalked = animWalked.has(i);
                if (isFrontier) {
                  opacity = 1;
                  stroke = palette.critical;
                  width = 3;
                  marker = "url(#arrC)";
                  filt = "url(#g2)";
                } else if (isWalked) {
                  opacity = 0.7;
                  stroke = palette.accent;
                  width = 1.8;
                  marker = "url(#arrL)";
                  filt = "url(#g1)";
                } else {
                  opacity = 0.08;
                  stroke = palette.edge;
                  width = 0.7;
                  marker = "url(#arr)";
                  filt = "none";
                }
              } else {
                const vis = hoveredNode === null || (connSet.has(e.from) && connSet.has(e.to));
                opacity = vis ? (isCrit ? 0.85 : 0.3) : 0.05;
                stroke = isCrit ? palette.critical : palette.accent;
                width = isCrit ? 2 : 1;
                marker = isCrit ? "url(#arrC)" : "url(#arr)";
                filt = isCrit ? "url(#g1)" : "none";
              }

              return <path key={`e${i}`}
                d={bezierPath(from.x + ox, from.y + oy, to.x - ox, to.y - oy)}
                fill="none" stroke={stroke} strokeWidth={width}
                opacity={opacity} markerEnd={marker} filter={filt}
                style={{ transition: "opacity 0.25s, stroke 0.25s, stroke-width 0.2s" }} />;
            })}

            {/* Nodes */}
            {dag.nodes.map(n => {
              const isCrit = showCritical && !animating && critPath.has(n.id);
              const isRoot = inDeg[n.id] === 0;
              const isLeaf = outDeg[n.id] === 0;

              let nOpacity, sColor, sW, glow, glowColor;
              if (animating) {
                const isActive = animActiveNode === n.id;
                const isVisited = animVisited.has(n.id);
                nOpacity = 1; // all nodes always visible
                if (isActive) {
                  sColor = palette.critical;
                  sW = 2.5;
                  glow = true;
                  glowColor = palette.critical;
                } else if (isVisited) {
                  sColor = palette.accent;
                  sW = 1.5;
                  glow = true;
                  glowColor = palette.accent;
                } else {
                  sColor = palette.edge;
                  sW = 1;
                  glow = false;
                  glowColor = palette.edge;
                }
              } else {
                const vis = hoveredNode === null || connSet.has(n.id);
                const hov = hoveredNode === n.id;
                nOpacity = vis ? 1 : 0.1;
                sColor = isCrit ? palette.critical : hov ? palette.accent : palette.edge;
                sW = isCrit ? 2 : hov ? 1.5 : 1;
                glow = isCrit || hov;
                glowColor = isCrit ? palette.critical : palette.accent;
              }

              return (
                <g key={n.id} style={{ opacity: nOpacity, transition: "opacity 0.25s" }}
                  onMouseEnter={() => !animating && setHoveredNode(n.id)}
                  onMouseLeave={() => !animating && setHoveredNode(null)}>

                  {glow && <>
                    <circle cx={n.x} cy={n.y} r={NODE_R + 8} fill="none"
                      stroke={glowColor}
                      strokeWidth="1" opacity="0.2" filter="url(#g2)" />
                    {animating && animActiveNode === n.id && (
                      <circle cx={n.x} cy={n.y} r={NODE_R + 16} fill="none"
                        stroke={palette.critical} strokeWidth="0.5" opacity="0.15" filter="url(#g3)" />
                    )}
                  </>}

                  <circle cx={n.x} cy={n.y} r={NODE_R}
                    fill={palette.node} stroke={sColor} strokeWidth={sW}
                    filter={glow ? "url(#g1)" : "none"}
                    style={{ transition: "stroke 0.25s, stroke-width 0.25s" }} />

                  {isRoot && <polygon
                    points={`${n.x},${n.y - 6} ${n.x + 5},${n.y + 3} ${n.x - 5},${n.y + 3}`}
                    fill="none" stroke={palette.accent} strokeWidth="1" opacity="0.5" />}
                  {isLeaf && <rect x={n.x - 4} y={n.y - 4} width={8} height={8}
                    fill="none" stroke={palette.accent} strokeWidth="1" opacity="0.5" rx="1" />}
                  {!isRoot && !isLeaf && <circle cx={n.x} cy={n.y} r={2.5}
                    fill="none" stroke={palette.accent} strokeWidth="0.8" opacity="0.35" />}

                  <text x={n.x} y={n.y + NODE_R + 13} textAnchor="middle" fontSize="8"
                    fill={animating ? (animActiveNode === n.id ? palette.critical : animVisited.has(n.id) ? palette.text : palette.text) : (isCrit ? palette.critical : palette.text)}
                    opacity={animating ? (animActiveNode === n.id ? 0.9 : animVisited.has(n.id) ? 0.5 : 0.25) : (nOpacity > 0.5 ? 0.55 : 0.1)}
                    fontFamily="'JetBrains Mono', monospace" fontWeight="300">
                    {n.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Stats */}
        <div className="stats">
          <span>nodes <span className="sv">{dag.nodes.length}</span></span>
          <span>edges <span className="sv">{dag.edges.length}</span></span>
          <span>critical <span className="sv">{critPath.size}</span></span>
          <span>roots <span className="sv">{dag.nodes.filter(n => inDeg[n.id] === 0).length}</span></span>
          <span>leaves <span className="sv">{dag.nodes.filter(n => outDeg[n.id] === 0).length}</span></span>
        </div>
        <div style={{ display: "flex", gap: 14, padding: "4px 18px", fontSize: 9, opacity: 0.3, letterSpacing: 1 }}>
          <span>△ root</span><span>□ leaf</span><span>○ mid</span>
          <span style={{ color: palette.critical }}>● critical</span>
        </div>
      </div>
    </div>
  );
}
