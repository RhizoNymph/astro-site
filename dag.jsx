import { useState, useRef, useCallback, useEffect, useMemo } from "react";

/* ─── Config ─── */
const WALK_SPEED = 800;
const HOLD_MS = 2200;
const FADE_MS = 900;
const BLANK_MS = 400;
const ROOT_SIDE = "bottom";
const PALETTE = {
  bg: "#060a10",
  node: "#0a1220",
  stroke: "#3399ff",
  accent: "#62a0ea",
  edge: "#813d9c",
  critical: "#dc8add",
};
const NODE_R = 20;
const W = 960, H = 600, PAD = 60;

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
function generateDAG(nodeCount, edgeDensity, seed) {
  const rng = mulberry32(seed);
  const nodes = [], edges = [];
  const layers = Math.max(3, Math.ceil(Math.sqrt(nodeCount) * 1.4));
  const perLayer = [];
  let rem = nodeCount;

  for (let i = 0; i < layers; i++) {
    const avg = rem / (layers - i);
    const c = i === layers - 1 ? rem : Math.max(1, Math.round(avg + (rng() - 0.5) * avg * 0.6));
    perLayer.push(Math.min(c, rem));
    rem -= perLayer[i];
    if (rem <= 0) break;
  }

  const actual = perLayer.filter(c => c > 0);
  let id = 0;

  actual.forEach((count, li) => {
    for (let i = 0; i < count; i++) {
      const layerT = actual.length === 1 ? 0.5 : li / (actual.length - 1);
      const slotT = count === 1 ? 0.5 : i / (count - 1);
      const rawY = PAD + layerT * (H - PAD * 2);
      const y = H - rawY + PAD + (rng() - 0.5) * 25;
      const x = PAD + slotT * (W - PAD * 2) + (rng() - 0.5) * 28;
      nodes.push({ id: id++, layer: li, x, y });
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
      const cands = nodes.filter(c => c.layer < n.layer);
      if (cands.length) {
        const p = cands[Math.floor(rng() * cands.length)];
        edges.push({ from: p.id, to: n.id });
      }
    }
  });

  return { nodes, edges };
}

/* ─── Topological Sort ─── */
function topoSort(nodes, edges) {
  const adj = {}, inD = {};
  nodes.forEach(n => { adj[n.id] = []; inD[n.id] = 0; });
  edges.forEach(e => { adj[e.from].push(e.to); inD[e.to]++; });
  const q = nodes.filter(n => inD[n.id] === 0).map(n => n.id);
  const order = [];
  while (q.length) {
    q.sort((a, b) => a - b);
    const u = q.shift();
    order.push(u);
    for (const v of adj[u]) { inD[v]--; if (inD[v] === 0) q.push(v); }
  }
  return order;
}

function bezierPath(x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  if (Math.abs(dx) > Math.abs(dy)) {
    return `M ${x1} ${y1} C ${x1 + dx * 0.4} ${y1}, ${x1 + dx * 0.6} ${y2}, ${x2} ${y2}`;
  }
  return `M ${x1} ${y1} C ${x1} ${y1 + dy * 0.4}, ${x2} ${y1 + dy * 0.6}, ${x2} ${y2}`;
}

function randRange(lo, hi) {
  return lo + Math.floor(Math.random() * (hi - lo + 1));
}

/* ─── Component ─── */
export default function DAGBackground() {
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 100000));
  const [nodeCount, setNodeCount] = useState(() => randRange(14, 28));
  const [density, setDensity] = useState(() => 0.3 + Math.random() * 0.4);
  const [visited, setVisited] = useState(new Set());
  const [walked, setWalked] = useState(new Set());
  const [frontier, setFrontier] = useState(new Set());
  const [activeNode, setActiveNode] = useState(null);
  const [fadeOp, setFadeOp] = useState(1);
  const timerRef = useRef(null);
  const cancelRef = useRef(false);

  const dag = useMemo(() => generateDAG(nodeCount, density, seed), [nodeCount, density, seed]);
  const nodeMap = useMemo(() => {
    const m = {}; dag.nodes.forEach(n => m[n.id] = n); return m;
  }, [dag]);

  const schedule = useCallback((fn, ms) => {
    return new Promise(res => {
      timerRef.current = setTimeout(() => {
        if (!cancelRef.current) { fn(); res(true); } else res(false);
      }, ms);
    });
  }, []);

  const runLoop = useCallback(async () => {
    cancelRef.current = false;
    let cSeed = seed;
    let cNodes = nodeCount;
    let cDens = density;

    while (!cancelRef.current) {
      const d = generateDAG(cNodes, cDens, cSeed);
      const order = topoSort(d.nodes, d.edges);

      setSeed(cSeed);
      setNodeCount(cNodes);
      setDensity(cDens);
      setVisited(new Set());
      setWalked(new Set());
      setFrontier(new Set());
      setActiveNode(null);
      setFadeOp(1);

      if (!(await schedule(() => {}, 600))) break;

      const vis = new Set();
      const wlk = new Set();
      let broke = false;

      for (let i = 0; i < order.length; i++) {
        if (cancelRef.current) { broke = true; break; }
        const nid = order[i];
        const front = new Set();
        d.edges.forEach((e, idx) => {
          if (e.to === nid && vis.has(e.from)) front.add(idx);
        });
        vis.add(nid);

        const sV = new Set(vis), sW = new Set(wlk), sF = new Set(front);
        if (!(await schedule(() => {
          setActiveNode(nid);
          setVisited(sV);
          setWalked(sW);
          setFrontier(sF);
        }, WALK_SPEED))) { broke = true; break; }

        front.forEach(idx => wlk.add(idx));
      }
      if (broke) break;

      if (!(await schedule(() => {
        setFrontier(new Set());
        setWalked(new Set(wlk));
        setActiveNode(null);
      }, WALK_SPEED))) break;

      if (!(await schedule(() => {}, HOLD_MS))) break;
      if (!(await schedule(() => setFadeOp(0), 50))) break;
      if (!(await schedule(() => {
        setVisited(new Set());
        setWalked(new Set());
        setFrontier(new Set());
        setActiveNode(null);
      }, FADE_MS))) break;

      cSeed = Math.floor(Math.random() * 100000);
      cNodes = randRange(14, 28);
      cDens = 0.3 + Math.random() * 0.4;

      if (!(await schedule(() => setFadeOp(1), BLANK_MS))) break;
    }
  }, []);

  useEffect(() => {
    const t = setTimeout(() => runLoop(), 200);
    return () => { cancelRef.current = true; clearTimeout(t); if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  const P = PALETTE;

  return (
    <div style={{
      width: "100%", height: "100%", background: P.bg,
      overflow: "hidden", position: "relative",
    }}>
      <div style={{
        opacity: fadeOp,
        transition: `opacity ${FADE_MS}ms ease-in-out`,
        width: "100%", height: "100%",
      }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="100%"
          preserveAspectRatio="xMidYMid slice"
          style={{ display: "block" }}>
          <defs>
            <filter id="g1"><feGaussianBlur stdDeviation="3" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            <filter id="g2"><feGaussianBlur stdDeviation="6" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            <filter id="g3"><feGaussianBlur stdDeviation="12" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill={P.accent} opacity="0.4" />
            </marker>
            <marker id="arrC" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill={P.critical} opacity="0.8" />
            </marker>
            <marker id="arrW" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill={P.accent} opacity="0.6" />
            </marker>
          </defs>

          {/* Subtle grid */}
          {Array.from({ length: 24 }, (_, i) =>
            Array.from({ length: 15 }, (_, j) => (
              <circle key={`${i}-${j}`} cx={i * 40 + 20} cy={j * 40 + 20}
                r="0.5" fill={P.edge} opacity="0.15" />
            ))
          )}

          {/* Edges */}
          {dag.edges.map((e, i) => {
            const from = nodeMap[e.from], to = nodeMap[e.to];
            const dx = to.x - from.x, dy = to.y - from.y;
            const d = Math.sqrt(dx * dx + dy * dy) || 1;
            const ox = (dx / d) * NODE_R, oy = (dy / d) * NODE_R;

            const isFront = frontier.has(i);
            const isWalk = walked.has(i);

            let stroke, width, opacity, marker, filt;
            if (isFront) {
              stroke = P.critical; width = 3; opacity = 1;
              marker = "url(#arrC)"; filt = "url(#g2)";
            } else if (isWalk) {
              stroke = P.accent; width = 1.8; opacity = 0.6;
              marker = "url(#arrW)"; filt = "url(#g1)";
            } else {
              stroke = P.edge; width = 0.7; opacity = 0.1;
              marker = "url(#arr)"; filt = "none";
            }

            return <path key={`e${i}`}
              d={bezierPath(from.x + ox, from.y + oy, to.x - ox, to.y - oy)}
              fill="none" stroke={stroke} strokeWidth={width}
              opacity={opacity} markerEnd={marker} filter={filt}
              style={{ transition: "opacity 0.3s, stroke 0.3s, stroke-width 0.25s" }} />;
          })}

          {/* Nodes */}
          {dag.nodes.map(n => {
            const isActive = activeNode === n.id;
            const isVis = visited.has(n.id);

            let sColor, sW, glow, glowColor;
            if (isActive) {
              sColor = P.critical; sW = 2.5; glow = true; glowColor = P.critical;
            } else if (isVis) {
              sColor = P.accent; sW = 1.5; glow = true; glowColor = P.accent;
            } else {
              sColor = P.edge; sW = 1; glow = false; glowColor = P.edge;
            }

            return (
              <g key={n.id} style={{ transition: "opacity 0.3s" }}>
                {glow && (
                  <circle cx={n.x} cy={n.y} r={NODE_R + 8} fill="none"
                    stroke={glowColor} strokeWidth="1" opacity="0.2" filter="url(#g2)" />
                )}
                {isActive && (
                  <circle cx={n.x} cy={n.y} r={NODE_R + 16} fill="none"
                    stroke={P.critical} strokeWidth="0.5" opacity="0.15" filter="url(#g3)" />
                )}

                <circle cx={n.x} cy={n.y} r={NODE_R}
                  fill={P.node} stroke={sColor} strokeWidth={sW}
                  filter={glow ? "url(#g1)" : "none"}
                  style={{ transition: "stroke 0.3s, stroke-width 0.25s" }} />

                {/* Inner glyph */}
                <circle cx={n.x} cy={n.y} r={2.5}
                  fill="none" stroke={isActive ? P.critical : isVis ? P.accent : P.stroke}
                  strokeWidth="0.8" opacity={isActive ? 0.7 : isVis ? 0.4 : 0.15}
                  style={{ transition: "stroke 0.3s, opacity 0.3s" }} />
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
