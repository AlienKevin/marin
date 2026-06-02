import { useState } from 'react'
import data from './phase05_data.json'

const TERM = {
  background: '#0d1117', color: '#e6edf3', borderRadius: 6, padding: '8px 10px',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
  fontSize: 11.5, lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
  overflow: 'auto', margin: 0, maxHeight: 200,
}

function degenerate(text) {
  // mirror the offline flag: long + highly repetitive
  if (text.length <= 800) return false
  // crude repetition proxy: unique-line ratio
  const lines = text.split('\n').filter((l) => l.trim())
  if (lines.length < 6) return false
  return new Set(lines).size / lines.length < 0.4
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
        {ep.transcript.map((t, j) => {
          const isAgent = t.role === 'assistant'
          const deg = !isAgent && degenerate(t.text)
          return (
            <div key={j} style={{ marginLeft: isAgent ? 0 : 24 }}>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.5px', opacity: 0.65, marginBottom: 3, color: isAgent ? '#d97706' : '#3b82f6' }}>
                {isAgent ? '▸ agent (command)' : '◂ simulator (observation)'} · turn {Math.floor(j / 2) + 1}
                {deg && <span style={{ color: '#dc2626', fontWeight: 700 }}> · ⚠ degenerate</span>}
              </div>
              <pre style={{ ...TERM, borderLeft: `3px solid ${isAgent ? '#d97706' : deg ? '#dc2626' : '#3b82f6'}` }}>
                {t.text}{t.trunc && <span style={{ opacity: 0.5 }}>{'\n'}… +{(t.len - t.text.length).toLocaleString()} more chars</span>}
              </pre>
            </div>
          )
        })}
      </div>
      <div style={{ fontSize: 11.5, opacity: 0.6, marginTop: 10 }}>
        Agent and simulator are the <em>same</em> (b) 1M network playing both roles. Full transcripts at{' '}
        <code>gs://marin-us-east5/closed-loop/b1m-selfplay-2/</code>.
      </div>
    </div>
  )
}
