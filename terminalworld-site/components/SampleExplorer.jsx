import { useState } from 'react'
import data from './sample_data.json'

const TERM = {
  background: '#0d1117',
  color: '#e6edf3',
  borderRadius: 8,
  padding: '10px 12px',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
  fontSize: 12,
  lineHeight: 1.5,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  overflow: 'auto',
  margin: 0,
}

const LABEL = {
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  opacity: 0.6,
  marginBottom: 4,
}

function Term({ text, accent, maxHeight = 260 }) {
  return (
    <pre style={{ ...TERM, maxHeight, borderLeft: accent ? `3px solid ${accent}` : 'none' }}>
      {text && text.length ? text : '(empty output)'}
    </pre>
  )
}

export function SampleExplorer() {
  const { models, trials } = data
  const [sel, setSel] = useState(() => {
    const d = models.filter((m) => m.default).map((m) => m.key)
    return (d.length ? d : models.map((m) => m.key)).slice(0, 2)
  })

  const toggle = (k) =>
    setSel((prev) =>
      prev.includes(k)
        ? prev.filter((x) => x !== k)
        : prev.length >= 2
        ? [prev[prev.length - 1], k]
        : [...prev, k]
    )

  const selModels = models.filter((m) => sel.includes(m.key))

  if (!trials || trials.length === 0) {
    return (
      <div style={{ opacity: 0.6, padding: '24px 0' }}>
        Samples are still generating — this page will populate once the sampling jobs finish.
      </div>
    )
  }

  return (
    <div>
      {/* sticky model selector */}
      <div
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 20,
          padding: '12px 0',
          marginBottom: 18,
          backdropFilter: 'blur(10px)',
          background: 'rgba(127,127,127,0.06)',
          borderBottom: '1px solid rgba(128,128,128,0.2)',
        }}
      >
        <div style={{ fontSize: 13, opacity: 0.7, marginBottom: 8 }}>
          Select 1–2 models to compare (median env-PPL shown):
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {models.map((m) => {
            const on = sel.includes(m.key)
            return (
              <button
                key={m.key}
                onClick={() => toggle(m.key)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 8,
                  cursor: 'pointer',
                  border: `1.5px solid ${m.color}`,
                  borderRadius: 999,
                  padding: '6px 14px',
                  background: on ? m.color : 'transparent',
                  color: on ? '#fff' : 'inherit',
                  fontSize: 13,
                  fontWeight: 600,
                  transition: 'all 0.15s ease',
                }}
              >
                <span
                  style={{
                    width: 9,
                    height: 9,
                    borderRadius: 99,
                    background: on ? '#fff' : m.color,
                  }}
                />
                {m.label}
                <span style={{ opacity: 0.7, fontWeight: 400 }}>· {m.ppl}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* trials */}
      {trials.map((t, i) => (
        <div
          key={t.id}
          style={{
            border: '1px solid rgba(128,128,128,0.25)',
            borderRadius: 12,
            padding: 16,
            marginBottom: 22,
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'baseline',
              flexWrap: 'wrap',
              gap: 8,
              marginBottom: 12,
            }}
          >
            <strong style={{ fontSize: 15 }}>
              #{i + 1} · {t.repo}
            </strong>
            <code style={{ fontSize: 11, opacity: 0.6 }}>{t.id}</code>
          </div>

          <div style={LABEL}>15th command (the input — same for every model)</div>
          <Term text={t.cmd} maxHeight={130} />

          <div style={{ height: 12 }} />
          <div style={{ ...LABEL, color: '#059669', fontWeight: 700, opacity: 1 }}>
            ✓ ground truth — real terminal output · {t.real.length} chars
          </div>
          <Term text={t.real} accent="#059669" />

          <div style={{ height: 16 }} />
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${Math.max(selModels.length, 1)}, minmax(0, 1fr))`,
              gap: 14,
            }}
          >
            {selModels.map((m) => (
              <div key={m.key} style={{ minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 10 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 99, background: m.color }} />
                  <strong style={{ fontSize: 13 }}>{m.label}</strong>
                  <span style={{ fontSize: 11, opacity: 0.6 }}>· PPL {m.ppl}</span>
                </div>
                {(t.samples[m.key] || []).map((s, j) => (
                  <div key={j} style={{ marginBottom: 9 }}>
                    <div style={{ ...LABEL, marginBottom: 3 }}>
                      sample {j + 1} · {s.length} ch
                    </div>
                    <Term text={s} accent={m.color} maxHeight={220} />
                  </div>
                ))}
                {!t.samples[m.key] && (
                  <div style={{ opacity: 0.5, fontSize: 12 }}>no samples for this model</div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
