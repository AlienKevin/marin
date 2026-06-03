import { useState } from 'react'
import data from './phase05_data.json'

const TERM = {
  background: '#0d1117', color: '#e6edf3', borderRadius: 6, padding: '8px 10px',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
  fontSize: 11.5, lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
  overflow: 'auto', margin: 0,
}

// Flag human-noticeable repetition (matches the 5.9% metric on this page), not
// just catastrophic >800-char loops — short exact-line loops like the 776-char
// `warmupIterations` ×9 observation were previously missed. Returns {kind, n, severe} | null.
function repetition(text) {
  const lines = text.split('\n').filter((l) => l.trim())
  if (lines.length < 4) return null
  let maxRun = 1, cur = 1
  for (let i = 1; i < lines.length; i++) {
    cur = lines[i] === lines[i - 1] ? cur + 1 : 1
    if (cur > maxRun) maxRun = cur
  }
  const dupRatio = 1 - new Set(lines).size / lines.length
  const parse = (s) => s.match(/^(.*?)(\d+)(.*)$/)
  let maxInc = 1, curInc = 1
  for (let i = 1; i < lines.length; i++) {
    const a = parse(lines[i - 1]), b = parse(lines[i])
    const ok = a && b && a[1] === b[1] && a[3] === b[3] && +b[2] === +a[2] + 1
    curInc = ok ? curInc + 1 : 1
    if (curInc > maxInc) maxInc = curInc
  }
  const severe = text.length > 800 && dupRatio >= 0.6 // the catastrophic 1.76% class
  if (maxInc >= 4) return { kind: 'incrementing', n: maxInc, severe }
  if (maxRun >= 3) return { kind: 'repeated line', n: maxRun, severe }
  if (dupRatio >= 0.4 && lines.length >= 5) return { kind: 'dup lines', n: Math.round(dupRatio * 100) + '%', severe }
  return null
}

function Turn({ t, idx }) {
  const [open, setOpen] = useState(false)
  const isAgent = t.role === 'assistant'
  const shown = open ? t.full : t.preview
  const rep = isAgent ? null : repetition(t.full)
  const canExpand = t.preview_trunc
  return (
    <div style={{ marginLeft: isAgent ? 0 : 24 }}>
      <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.5px', opacity: 0.65, marginBottom: 3, color: isAgent ? '#d97706' : '#3b82f6' }}>
        {isAgent ? '▸ agent (command)' : '◂ simulator (observation)'} · turn {Math.floor(idx / 2) + 1}
        {rep && <span style={{ color: rep.severe ? '#dc2626' : '#ea580c', fontWeight: 700 }}> · ⚠ {rep.kind} ×{rep.n}{rep.severe ? ' · runaway' : ''}</span>}
        <span style={{ opacity: 0.55, fontWeight: 400 }}> · {t.len.toLocaleString()} chars</span>
      </div>
      <pre style={{ ...TERM, maxHeight: open ? 'none' : 220, borderLeft: `3px solid ${isAgent ? '#d97706' : rep ? (rep.severe ? '#dc2626' : '#ea580c') : '#3b82f6'}` }}>
        {shown}
        {open && t.full_trunc && <span style={{ opacity: 0.5 }}>{'\n'}… +{(t.len - t.full.length).toLocaleString()} more chars (truncated in data export)</span>}
      </pre>
      {canExpand && (
        <button onClick={() => setOpen((v) => !v)} style={{
          marginTop: 4, cursor: 'pointer', border: 'none', background: 'transparent',
          color: isAgent ? '#d97706' : '#3b82f6', fontSize: 11, fontWeight: 600, padding: '2px 0',
        }}>
          {open ? '▲ collapse' : `▼ expand full observation (${t.len.toLocaleString()} chars)`}
        </button>
      )}
    </div>
  )
}

export function Phase05Browser() {
  const { episodes } = data
  const [sel, setSel] = useState(0)
  const ep = episodes[sel]
  return (
    <div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
        {episodes.map((e, i) => (
          <button key={e.id} onClick={() => setSel(i)} style={{
            cursor: 'pointer', borderRadius: 8, padding: '6px 12px', fontSize: 12.5,
            border: `1.5px solid ${i === sel ? '#059669' : 'rgba(128,128,128,0.4)'}`,
            background: i === sel ? '#059669' : 'transparent', color: i === sel ? '#fff' : 'inherit',
            fontWeight: i === sel ? 700 : 400,
          }}>{e.label}</button>
        ))}
      </div>

      <div style={{ fontSize: 12.5, opacity: 0.8, marginBottom: 10 }}>
        <strong>{ep.repo}</strong> · {ep.turns_total} turns · {ep.clean ? 'clean termination ✓' : 'ran to turn cap'} ·{' '}
        {ep.n_sim_degen === 0 ? 'no degenerate sim turns' : `${ep.n_sim_degen} degenerate sim turn(s)`} ·{' '}
        <span style={{ opacity: 0.6 }}>showing first {ep.shown} of {ep.recorded} turns</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {ep.transcript.map((t, j) => <Turn key={j} t={t} idx={j} />)}
      </div>
      <div style={{ fontSize: 11.5, opacity: 0.6, marginTop: 10 }}>
        Agent and simulator are the <em>same</em> (b) 1M network playing both roles. Click <strong>expand</strong> to see a full observation. Complete transcripts at{' '}
        <code>gs://marin-us-east5/closed-loop/b1m-selfplay-2/</code>.
      </div>
    </div>
  )
}
