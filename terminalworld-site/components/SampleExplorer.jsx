import { useState } from 'react'
import data from './sample_data.json'

const TERM = {
  background: '#0d1117', color: '#e6edf3', borderRadius: 8, padding: '9px 11px',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
  fontSize: 12, lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
  overflow: 'auto', margin: 0,
}
const LABEL = { fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.5px', opacity: 0.6, marginBottom: 4 }

function Dot({ c }) {
  return <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 99, background: c }} />
}

function Term({ field, accent, maxHeight = 260 }) {
  const empty = !field.text || field.text.length === 0
  return (
    <pre style={{ ...TERM, maxHeight, borderLeft: accent ? `3px solid ${accent}` : 'none' }}>
      {empty ? '(empty output)' : field.text}
      {field.trunc && <span style={{ opacity: 0.5 }}>{'\n'}… +{(field.len - field.text.length).toLocaleString()} more chars</span>}
    </pre>
  )
}

// Grouped bar chart: x = difficulty bucket, bars = models, height = median env-PPL.
// Color encodes *model identity only* (matches the selector pills); the numbers
// in the table stay black.
function BarChart({ models, buckets, bucketSummary, bucketOrder }) {
  const W = 780, H = 300
  const mL = 30, mR = 10, mT = 14, mB = 70
  const pw = W - mL - mR, ph = H - mT - mB
  const YMAX = 10.5
  const y = (v) => mT + ph * (1 - Math.min(v, YMAX) / YMAX)
  const base = mT + ph
  const gw = pw / bucketOrder.length
  const pad = 9
  const bw = (gw - pad * 2) / models.length
  const bmeta = Object.fromEntries(buckets.map((b) => [b.key, b]))
  const grid = [1, 2, 4, 6, 8, 10]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }} role="img"
         aria-label="median env-PPL by difficulty bucket and model">
      {grid.map((g) => (
        <g key={g}>
          <line x1={mL} x2={W - mR} y1={y(g)} y2={y(g)} stroke="rgba(128,128,128,0.18)" strokeWidth="1" />
          <text x={mL - 5} y={y(g) + 3} fontSize="9" textAnchor="end" fill="rgba(128,128,128,0.75)">{g}</text>
        </g>
      ))}
      {/* "perfect" reference at PPL = 1.0 */}
      <line x1={mL} x2={W - mR} y1={y(1)} y2={y(1)} stroke="#059669" strokeDasharray="4 3" strokeWidth="1" />
      <text x={W - mR} y={y(1) - 3} fontSize="8.5" textAnchor="end" fill="#059669">perfect = 1.0</text>
      {bucketOrder.map((bk, gi) => {
        const gx = mL + gi * gw
        return (
          <g key={bk}>
            {models.map((m, mi) => {
              const v = bucketSummary[bk][m.key]
              const bx = gx + pad + mi * bw
              const capped = v > YMAX
              return (
                <g key={m.key}>
                  <rect x={bx} y={y(v)} width={Math.max(bw - 1.5, 1)} height={base - y(v)} fill={m.color} opacity="0.92">
                    <title>{`${m.label} · ${bmeta[bk].label}: ${v.toFixed(2)}`}</title>
                  </rect>
                  {capped && <text x={bx + bw / 2} y={y(v) - 2} fontSize="7" textAnchor="middle" fill="currentColor">{v.toFixed(0)}</text>}
                </g>
              )
            })}
            <text x={gx + gw / 2} y={base + 15} fontSize="9.5" textAnchor="middle" fill="currentColor">
              {bmeta[bk].label.split(' (')[0]}
            </text>
            <text x={gx + gw / 2} y={base + 27} fontSize="8" textAnchor="middle" fill="rgba(128,128,128,0.7)">
              n={bmeta[bk].counts}
            </text>
          </g>
        )
      })}
      <text x={mL - 26} y={mT - 3} fontSize="9" fill="rgba(128,128,128,0.8)">PPL</text>
    </svg>
  )
}

export function SampleExplorer() {
  const { models, buckets, bucketSummary, bucketOrder, trials } = data
  const [sel, setSel] = useState(['a-1m', 'b-1m'])
  const [bucketFilter, setBucketFilter] = useState('all')

  const toggle = (k) =>
    setSel((prev) =>
      prev.includes(k)
        ? (prev.length === 1 ? prev : prev.filter((x) => x !== k))
        : (prev.length >= 2 ? [prev[prev.length - 1], k] : [...prev, k]))

  const selModels = models.filter((m) => sel.includes(m.key))
  const shown = trials.filter((t) => bucketFilter === 'all' || t.bucket === bucketFilter)
  const bmeta = Object.fromEntries(buckets.map((b) => [b.key, b]))

  if (!trials || trials.length === 0)
    return <div style={{ opacity: 0.6, padding: '24px 0' }}>Samples still generating — this page will populate when the jobs finish.</div>

  return (
    <div>
      {/* monochrome summary table — numbers stay black */}
      <div style={{ overflowX: 'auto', marginBottom: 6 }}>
        <table style={{ fontSize: 13, borderCollapse: 'collapse', minWidth: 540 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '4px 10px 6px 0' }}>median env-PPL ↓</th>
              {models.map((m) => (
                <th key={m.key} style={{ padding: '4px 8px 6px', textAlign: 'right', whiteSpace: 'nowrap', fontWeight: 600 }}>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, justifyContent: 'flex-end' }}>
                    <Dot c={m.color} />{m.label}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {bucketOrder.map((bk) => (
              <tr key={bk} style={{ borderTop: '1px solid rgba(128,128,128,0.2)' }}>
                <td style={{ padding: '4px 10px 4px 0', whiteSpace: 'nowrap' }}>{bmeta[bk].label} <span style={{ opacity: 0.5 }}>({bmeta[bk].counts})</span></td>
                {models.map((m) => {
                  const v = bucketSummary[bk][m.key]
                  return <td key={m.key} style={{ padding: '4px 8px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{v?.toFixed(2)}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* bar-chart visualization (color = model identity only) */}
      <div style={{ border: '1px solid rgba(128,128,128,0.2)', borderRadius: 10, padding: '10px 12px 4px', marginBottom: 20 }}>
        <BarChart models={models} buckets={buckets} bucketSummary={bucketSummary} bucketOrder={bucketOrder} />
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center', fontSize: 11.5, marginTop: 4 }}>
          {models.map((m) => (
            <span key={m.key} style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}><Dot c={m.color} />{m.label}</span>
          ))}
        </div>
        <div style={{ fontSize: 11.5, opacity: 0.6, textAlign: 'center', marginTop: 4 }}>
          Bars near the dashed line (1.0) are faithful. Buckets run easy→hard; only the hard buckets separate the arms.
        </div>
      </div>

      {/* sticky controls */}
      <div style={{ position: 'sticky', top: 0, zIndex: 20, padding: '10px 0', marginBottom: 16,
                    backdropFilter: 'blur(10px)', background: 'rgba(127,127,127,0.06)', borderBottom: '1px solid rgba(128,128,128,0.2)' }}>
        <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>Compare 1–2 models:</div>
        <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap', marginBottom: 10 }}>
          {models.map((m) => {
            const on = sel.includes(m.key)
            return (
              <button key={m.key} onClick={() => toggle(m.key)} style={{
                display: 'inline-flex', alignItems: 'center', gap: 7, cursor: 'pointer',
                border: `1.5px solid ${m.color}`, borderRadius: 999, padding: '5px 12px',
                background: on ? m.color : 'transparent', color: on ? '#fff' : 'inherit', fontSize: 12.5, fontWeight: 600 }}>
                <span style={{ width: 8, height: 8, borderRadius: 99, background: on ? '#fff' : m.color }} />
                {m.label}
              </button>
            )
          })}
        </div>
        <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>Filter by regime:</div>
        <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
          {[{ key: 'all', label: `All (${trials.length})` }, ...buckets.map((b) => ({ key: b.key, label: `${b.label} (${b.counts})` }))].map((b) => {
            const on = bucketFilter === b.key
            return (
              <button key={b.key} onClick={() => setBucketFilter(b.key)} style={{
                cursor: 'pointer', border: '1px solid rgba(128,128,128,0.4)', borderRadius: 6, padding: '4px 10px',
                background: on ? 'rgba(128,128,128,0.25)' : 'transparent', fontSize: 12, fontWeight: on ? 700 : 400 }}>
                {b.label}
              </button>
            )
          })}
        </div>
      </div>

      {/* trials */}
      {shown.map((t, i) => {
        const showHeader = i === 0 || shown[i - 1].bucket !== t.bucket
        return (
          <div key={t.id}>
            {showHeader && (
              <div style={{ margin: '18px 0 10px', paddingBottom: 4, borderBottom: '2px solid rgba(128,128,128,0.3)' }}>
                <strong style={{ fontSize: 15 }}>{bmeta[t.bucket].label}</strong>
                <div style={{ fontSize: 12.5, opacity: 0.7, marginTop: 2 }}>{bmeta[t.bucket].desc}</div>
              </div>
            )}
            <div style={{ border: '1px solid rgba(128,128,128,0.25)', borderRadius: 12, padding: 15, marginBottom: 18 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
                <strong style={{ fontSize: 14 }}>{t.repo}</strong>
                <span style={{ fontSize: 11, opacity: 0.6 }}>
                  {t.empty ? 'empty output' : `${Math.round((t.copyable || 0) * 100)}% copyable`} · <code>{t.id}</code>
                </span>
              </div>

              <div style={LABEL}>15th command</div>
              <Term field={t.cmd} maxHeight={120} />
              <div style={{ height: 10 }} />
              <div style={{ ...LABEL, color: '#059669', fontWeight: 700, opacity: 1 }}>✓ ground truth · {t.real.len.toLocaleString()} chars</div>
              <Term field={t.real} accent="#059669" />

              <div style={{ height: 14 }} />
              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.max(selModels.length, 1)}, minmax(0,1fr))`, gap: 12 }}>
                {selModels.map((m) => {
                  const c = t.cells[m.key]
                  return (
                    <div key={m.key} style={{ minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 7, flexWrap: 'wrap' }}>
                        <Dot c={m.color} />
                        <strong style={{ fontSize: 12.5 }}>{m.label}</strong>
                        <span style={{ fontSize: 11.5, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>
                          env-PPL {c?.ppl != null ? c.ppl.toFixed(2) : '—'}
                        </span>
                      </div>
                      {c ? (
                        <>
                          <Term field={c} accent={m.color} maxHeight={260} />
                          <div style={{ fontSize: 11, opacity: 0.7, marginTop: 5 }}>
                            representative of 5 · mean similarity {c.mean_sim?.toFixed(2)}
                          </div>
                          <details style={{ marginTop: 5 }}>
                            <summary style={{ fontSize: 11, cursor: 'pointer', opacity: 0.7 }}>all 5 samples</summary>
                            <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 6 }}>
                              {c.all.map((s, j) => <Term key={j} field={s} maxHeight={150} />)}
                            </div>
                          </details>
                        </>
                      ) : <div style={{ opacity: 0.5, fontSize: 12 }}>no samples yet</div>}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )
      })}
      <div style={{ fontSize: 12.5, opacity: 0.7, marginTop: 10 }}>
        {trials.length} held-out trials, stratified by how much of the 15th observation is verbatim-recoverable from context. Each model sampled 5× per trial at temperature 1.0. env-PPL is the teacher-forced perplexity of the <em>real</em> observation under that model (per-trial, not the corpus mean).
      </div>
    </div>
  )
}
