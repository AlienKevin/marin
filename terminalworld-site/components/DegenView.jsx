import data from './degen_data.json'

const TERM = {
  background: '#0d1117', color: '#e6edf3', borderRadius: 8, padding: '10px 12px',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
  fontSize: 12, lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
  overflow: 'auto', margin: 0, maxHeight: 320,
}

function Bar({ pct, color, max = 100 }) {
  return (
    <div style={{ background: 'rgba(128,128,128,0.15)', borderRadius: 4, height: 16, position: 'relative', minWidth: 80 }}>
      <div style={{ width: `${(pct / max) * 100}%`, background: color, height: '100%', borderRadius: 4 }} />
    </div>
  )
}

export function DegenByModel() {
  const m = data.byModel
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(120px,180px) 1fr 48px', gap: '6px 12px', alignItems: 'center', margin: '12px 0' }}>
      {m.map((x) => (
        <>
          <div key={x.key + 'l'} style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 9, height: 9, borderRadius: 99, background: x.color }} />{x.label}
          </div>
          <div key={x.key + 'b'}><Bar pct={x.degen_pct} color={x.color} /></div>
          <div key={x.key + 'v'} style={{ fontVariantNumeric: 'tabular-nums', fontSize: 13, fontWeight: 600 }}>{x.degen_pct}%</div>
        </>
      ))}
    </div>
  )
}

export function DegenByBand() {
  const b = data.bands
  const labels = { 'ppl<1.5': 'PPL < 1.5 (faithful)', '1.5-3': 'PPL 1.5–3', '3-10': 'PPL 3–10', 'ppl>10': 'PPL > 10 (bad)' }
  const max = Math.max(...b.map((x) => x.degen_pct))
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(120px,170px) 1fr 70px', gap: '6px 12px', alignItems: 'center', margin: '12px 0' }}>
      {b.map((x) => (
        <>
          <div key={x.band + 'l'} style={{ fontSize: 13 }}>{labels[x.band]}</div>
          <div key={x.band + 'b'}><Bar pct={x.degen_pct} color="#dc2626" max={max} /></div>
          <div key={x.band + 'v'} style={{ fontVariantNumeric: 'tabular-nums', fontSize: 13, fontWeight: 600 }}>{x.degen_pct}% <span style={{ opacity: 0.5, fontWeight: 400 }}>({x.degen}/{x.n})</span></div>
        </>
      ))}
    </div>
  )
}

export function DegenExemplar() {
  const e = data.exemplar
  return (
    <div>
      <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 5 }}>
        {e.repo} · (b) full-transcript 1M · teacher-forced env-PPL <strong>{e.ppl}</strong> · 1 of 5 sampled draws (truncated):
      </div>
      <pre style={TERM}>{e.head}{'\n'}<span style={{ opacity: 0.5 }}>… continues incrementing analysis_module_N to ~20,000 chars (the 4096-token cap)</span></pre>
    </div>
  )
}
