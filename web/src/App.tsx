import { useCallback, useEffect, useMemo, useState } from 'react'
import './App.css'

type Row = {
  symbol: string
  risk_score: number
  max_point_gap_next_pct: number
  max_q90_proxy_pct: number
  risk_score_down?: number | null
  min_point_gap_next_pct?: number | null
  min_q10_proxy_pct?: number | null
  context_days: number
  last_date: string
  last_close?: number | null
  implied_target_px?: number | null
  implied_down_target_px?: number | null
  forward_sessions?: number | null
  ranking_at?: string | null
  ml_probability?: number | null
  ml_skip_reason?: string | null
}

type ErrRow = { symbol: string; error: string }

/** Set by GET /api/latest — drives banners without guessing from row shape alone. */
type EnrichStatus = 'complete' | 'full_pending' | 'light' | 'prices_incomplete'

type BatchData = {
  /** 2 = gap-down fields, prices, ranking_at per row; omit/1 = legacy file */
  batch_schema_version?: number
  enrich_status?: EnrichStatus
  /** Server-side full batch (batch_rank) in progress */
  rebuild_ranking_running?: boolean
  /** API is filling gap-down via TimesFM in the background */
  pending_timesfm_enrich?: boolean
  generated_at: string
  params: { years: number; gap_threshold_pct: number; forward_trading_days: number; max_context_days: number }
  total_symbols: number
  rows: Row[]
  errors: ErrRow[]
}

/** True when the file on disk is too old to show prices (no Parquet cache / enrich failed). */
function isStaleBatchFile(b: BatchData | null): boolean {
  if (!b?.rows?.length) return false
  if (b.enrich_status === 'prices_incomplete') return true
  if (b.enrich_status !== undefined && b.enrich_status !== 'prices_incomplete') return false
  if (b.batch_schema_version != null && b.batch_schema_version >= 2) return false
  const r = b.rows[0]
  return r.last_close === undefined
}

/** Gap-down columns still empty; show info banner (legacy API infers from rows). */
function gapDownInfoBannerShown(b: BatchData | null): boolean {
  if (!b?.rows?.length) return false
  if (b.enrich_status === 'full_pending' || b.enrich_status === 'light') return true
  if (b.enrich_status === 'complete' || b.enrich_status === 'prices_incomplete') return false
  if (b.batch_schema_version != null && b.batch_schema_version >= 2) return false
  const r = b.rows[0]
  return !!(b.pending_timesfm_enrich || (r.last_close != null && r.risk_score_down === undefined))
}

function enrichBatchData(d: BatchData): BatchData {
  return {
    ...d,
    rows: d.rows.map((r) => ({
      ...r,
      ranking_at: r.ranking_at ?? d.generated_at,
      forward_sessions: r.forward_sessions ?? d.params.forward_trading_days,
    })),
  }
}

type SortKeyUp =
  | 'symbol'
  | 'risk_score'
  | 'max_point_gap_next_pct'
  | 'max_q90_proxy_pct'
  | 'ml_probability'
  | 'last_date'
  | 'last_close'
  | 'implied_target_px'
  | 'forward_sessions'
  | 'ranking_at'

type SortKeyDown =
  | 'symbol'
  | 'risk_score_down'
  | 'min_point_gap_next_pct'
  | 'min_q10_proxy_pct'
  | 'ml_probability'
  | 'last_date'
  | 'last_close'
  | 'implied_down_target_px'
  | 'forward_sessions'
  | 'ranking_at'

const DEFAULT_SORT_UP: { key: SortKeyUp; dir: 'asc' | 'desc' } = { key: 'risk_score', dir: 'desc' }
const DEFAULT_SORT_DOWN: { key: SortKeyDown; dir: 'asc' | 'desc' } = { key: 'risk_score_down', dir: 'desc' }

const TT_UP = {
  symbol: 'Ticker. Importance: 2/10.',
  riskScore:
    'Upside risk 0–100 vs threshold from TimesFM (larger positive overnight gap tail). Importance: 9/10.',
  maxPt: 'Max point forecast gap % (close→open) in horizon. Importance: 10/10.',
  q90: 'Upper quantile gap % (uncalibrated). Importance: 5/10.',
  ml: 'ML P(large upside gap) if model trained. Importance: 7/10.',
  lastDate: 'Last daily bar date (Alpaca). Importance: 6/10.',
  lastClose: 'Last regular close. Importance: 8/10.',
  tgt: 'Implied next open if max point gap realized: close×(1+maxPt%). Importance: 8/10.',
  within: 'Trading sessions in forecast window. Importance: 7/10.',
  scored: 'When this row was scored. Importance: 6/10.',
} as const

const TT_DN = {
  symbol: 'Ticker. Importance: 2/10.',
  riskDown:
    'Downside risk 0–100 vs threshold from TimesFM (large negative overnight gap tail). Importance: 9/10.',
  minPt: 'Min point forecast gap % (most bearish session in horizon). Importance: 10/10.',
  q10: 'Lower quantile gap % (uncalibrated). Importance: 5/10.',
  ml: 'Same ML as up table (trained on upside events)—directional reference only. Importance: 4/10.',
  lastDate: 'Last daily bar date (Alpaca). Importance: 6/10.',
  lastClose: 'Last regular close. Importance: 8/10.',
  tgt: 'Implied next open if min point gap realized: close×(1+minPt%). Importance: 8/10.',
  within: 'Trading sessions in forecast window. Importance: 7/10.',
  scored: 'When this row was scored. Importance: 6/10.',
} as const

export default function App() {
  const [dataTab, setDataTab] = useState<'batch' | 'custom'>('batch')

  const [batch, setBatch] = useState<BatchData | null>(null)
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchErr, setBatchErr] = useState<string | null>(null)
  const [rebuilding, setRebuilding] = useState(false)
  const [search, setSearch] = useState('')
  const [minScore, setMinScore] = useState(0)

  const [symbolsText, setSymbolsText] = useState('AAPL\nMSFT\nNVDA')
  const [years, setYears] = useState(5)
  const [threshold, setThreshold] = useState(10)
  const [forwardDays, setForwardDays] = useState(5)
  const [maxContext, setMaxContext] = useState(512)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{ rows: Row[]; errors: ErrRow[] } | null>(null)

  const [apiHealth, setApiHealth] = useState<{
    loaded: boolean
    rebuildAvailable: boolean
    batchScriptMissing?: boolean
    scheduler?: {
      enabled: boolean
      schedule: string
      next_run: string | null
      currently_running: boolean
    }
  }>({ loaded: false, rebuildAvailable: true })

  const probeHealth = useCallback(async () => {
    try {
      const r = await fetch('/api/health')
      if (!r.ok) throw new Error('bad health')
      const h: {
        features?: { rebuild_ranking?: boolean; batch_script_present?: boolean; nightly_scheduler?: boolean }
        scheduler?: { enabled: boolean; schedule: string; next_run: string | null; currently_running: boolean }
      } = await r.json()
      const hasRoute = h.features?.rebuild_ranking === true
      const scriptOk = h.features?.batch_script_present === true
      setApiHealth({
        loaded: true,
        rebuildAvailable: hasRoute && scriptOk,
        batchScriptMissing: hasRoute && !scriptOk,
        scheduler: h.scheduler,
      })
    } catch {
      setApiHealth({ loaded: true, rebuildAvailable: false })
    }
  }, [])

  const loadBatch = useCallback(
    async (opts?: { silent?: boolean }) => {
      setBatchLoading(true)
      if (!opts?.silent) setBatchErr(null)
      try {
        const res = await fetch('/api/latest')
        if (!res.ok) throw new Error(await res.text())
        const d: BatchData = enrichBatchData(await res.json())
        setBatch(d)
      } catch (e) {
        if (!opts?.silent) {
          setBatchErr(e instanceof Error ? e.message : 'Failed')
        }
      } finally {
        setBatchLoading(false)
      }
      if (!opts?.silent) void probeHealth()
    },
    [probeHealth]
  )

  const startRebuild = useCallback(async () => {
    setBatchErr(null)
    try {
      const res = await fetch('/api/rebuild-ranking', { method: 'POST' })
      if (res.status === 404) {
        setBatchErr(
          'The API you are running does not expose “Refresh rankings” yet. Stop uvicorn and start it again from the project (same command you use for /api/latest), ideally with --reload, then try once more.'
        )
        return
      }
      if (res.status === 409) {
        setBatchErr('A refresh is already in progress. We will keep checking for updates.')
        setRebuilding(true)
        window.setTimeout(() => loadBatch({ silent: true }), 4000)
        return
      }
      if (!res.ok) {
        const t = await res.text()
        throw new Error(t || res.statusText)
      }
      setRebuilding(true)
      window.setTimeout(() => loadBatch({ silent: true }), 4000)
    } catch (e) {
      setBatchErr(e instanceof Error ? e.message : 'Could not start refresh')
    }
  }, [loadBatch])

  useEffect(() => {
    if (!rebuilding) return
    const id = window.setInterval(() => {
      loadBatch({ silent: true })
    }, 10000)
    return () => clearInterval(id)
  }, [rebuilding, loadBatch])

  useEffect(() => {
    if (batch && rebuilding && !isStaleBatchFile(batch)) {
      setRebuilding(false)
    }
  }, [batch, rebuilding])

  useEffect(() => {
    let cancelled = false
    if (dataTab !== 'batch') return
    fetch('/api/rebuild-ranking/status')
      .then((r) => r.json())
      .then((s: { running?: boolean }) => {
        if (!cancelled && s.running) setRebuilding(true)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [dataTab])

  useEffect(() => {
    void probeHealth()
  }, [probeHealth])

  useEffect(() => {
    loadBatch()
  }, [loadBatch])

  const filtered = batch
    ? batch.rows.filter(
        (r) =>
          r.risk_score >= minScore &&
          (search === '' || r.symbol.toUpperCase().includes(search.toUpperCase()))
      )
    : []

  const runCustom = useCallback(async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch('/api/rank', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbols_text: symbolsText,
          years,
          gap_threshold_pct: threshold,
          forward_trading_days: forwardDays,
          max_context_days: maxContext,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      setResult(await res.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [symbolsText, years, threshold, forwardDays, maxContext])

  const activeRows = dataTab === 'batch' ? filtered : result?.rows ?? []
  const activeErrors = dataTab === 'batch' ? batch?.errors ?? [] : result?.errors ?? []
  const batchParams = batch?.params
  const showMain = dataTab === 'batch' ? batch : result

  return (
    <div className="app">
      <header className="shell-header">
        <div className="shell-inner">
          <div className="brand">
            <div className="brand-mark">OL</div>
            <div className="brand-text">
              <h1>Overnight Lab</h1>
              <p>
                Institutional-style overnight gap intelligence — prior <strong>close</strong> to next{' '}
                <strong>open</strong>, TimesFM forecasts plus optional LightGBM (realized bars only).
              </p>
            </div>
          </div>
          <div className="pill-row">
            <button type="button" className={'pill' + (dataTab === 'batch' ? ' active' : '')} onClick={() => setDataTab('batch')}>
              Batch run
              {batch ? ` · ${batch.rows.length}` : ''}
            </button>
            <button
              type="button"
              className={'pill down' + (dataTab === 'custom' ? ' active down' : '')}
              onClick={() => setDataTab('custom')}
            >
              Custom universe
            </button>
          </div>
        </div>
      </header>

      <div className="shell-body">
        <aside className="sidebar-card">
          <h3>{dataTab === 'batch' ? 'Filters' : 'Parameters'}</h3>
          {dataTab === 'batch' ? (
            <>
              <label>
                Search symbol
                <input
                  type="text"
                  placeholder="Ticker…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  autoComplete="off"
                />
              </label>
              <label>
                Min upside risk score
                <input type="number" min={0} step={1} value={minScore} onChange={(e) => setMinScore(Number(e.target.value))} />
              </label>
              {batch && (
                <div className="meta-block">
                  <strong>{batch.total_symbols}</strong> symbols in file · <strong>{batch.rows.length}</strong> ranked
                  <br />
                  {batch.params.gap_threshold_pct}% threshold · {batch.params.forward_trading_days}d window ·{' '}
                  {batch.params.years}yr history
                  <br />
                  <span style={{ opacity: 0.85 }}>File: {new Date(batch.generated_at).toLocaleString()}</span>
                </div>
              )}
              <button type="button" className="btn btn-secondary" onClick={() => void loadBatch()} disabled={batchLoading}>
                {batchLoading ? 'Refreshing…' : 'Reload batch'}
              </button>
              {apiHealth.scheduler && (
                <div className="scheduler-block">
                  <div className="scheduler-title">
                    <span className={`scheduler-dot ${apiHealth.scheduler.enabled ? 'on' : 'off'}`} />
                    Nightly auto-refresh
                  </div>
                  <div className="scheduler-detail">
                    {apiHealth.scheduler.schedule}
                  </div>
                  {apiHealth.scheduler.next_run && (
                    <div className="scheduler-detail">
                      Next: {new Date(apiHealth.scheduler.next_run).toLocaleString()}
                    </div>
                  )}
                  {apiHealth.scheduler.currently_running && (
                    <div className="scheduler-detail scheduler-running">Running now…</div>
                  )}
                  {batch && (
                    <div className="scheduler-detail">
                      Last data: {new Date(batch.generated_at).toLocaleString()}
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <>
              <label>
                Symbols
                <textarea value={symbolsText} onChange={(e) => setSymbolsText(e.target.value)} spellCheck={false} rows={7} />
              </label>
              <div className="row2">
                <label>
                  Years
                  <input type="number" min={1} max={20} value={years} onChange={(e) => setYears(+e.target.value)} />
                </label>
                <label>
                  Gap threshold %
                  <input type="number" min={0.1} step={0.5} value={threshold} onChange={(e) => setThreshold(+e.target.value)} />
                </label>
              </div>
              <div className="row2">
                <label>
                  Forward days
                  <input type="number" min={1} max={20} value={forwardDays} onChange={(e) => setForwardDays(+e.target.value)} />
                </label>
                <label>
                  Max context
                  <input type="number" min={64} max={512} step={32} value={maxContext} onChange={(e) => setMaxContext(+e.target.value)} />
                </label>
              </div>
              <button type="button" className="btn btn-primary" disabled={loading} onClick={runCustom}>
                {loading ? 'Running…' : 'Run scan'}
              </button>
            </>
          )}
        </aside>

        <main className="main-col">
          {dataTab === 'batch' && batchErr && <div className="banner-error">{batchErr}</div>}
          {dataTab === 'batch' && batch && gapDownInfoBannerShown(batch) && (
            <div className="banner-info">
              <strong>
                {batch.enrich_status === 'light' && batch.rebuild_ranking_running
                  ? 'Full batch refresh running'
                  : batch.enrich_status === 'light'
                    ? 'Gap-down data not ready'
                    : 'Gap-down board updating'}
              </strong>
              <span>
                {batch.enrich_status === 'full_pending' || batch.pending_timesfm_enrich ? (
                  <>
                    The server is recomputing downside scores with TimesFM for every symbol (this can take several minutes).
                    Gap-up prices and ML (if trained) should already be visible. Reload periodically to pick up gap-down
                    columns—or use <strong>Refresh rankings</strong> for a full batch rewrite.
                  </>
                ) : batch.rebuild_ranking_running ? (
                  <>
                    A full server batch is rewriting <code className="inline-code">latest_ranking.json</code>. Gap-down
                    columns will appear when it finishes. This tab can stay open; use <strong>Reload batch</strong> or wait
                    for auto-refresh.
                  </>
                ) : (
                  <>
                    Gap-down forecasts are still missing (background TimesFM enrich did not start or finished with gaps).
                    Try <strong>Reload batch</strong>, ensure <code className="inline-code">OVERNIGHT_DISABLE_BG_ENRICH</code>{' '}
                    is not set, or run <strong>Refresh rankings</strong>.
                  </>
                )}
              </span>
            </div>
          )}
          {dataTab === 'batch' && batch && isStaleBatchFile(batch) && (
            <div className="banner-stale">
              <div className="banner-stale-text">
                <strong>Update your saved rankings</strong>
                <span>
                  A newer layout includes gap-down scores, live prices, and ML columns. Refresh runs on the server and may take
                  several minutes—leave this tab open. Custom scan always uses the latest engine and does not need this step.
                </span>
                {apiHealth.loaded && !apiHealth.rebuildAvailable && (
                  <span className="banner-stale-footnote">
                    {apiHealth.batchScriptMissing ? (
                      <>This server install is missing the batch script. </>
                    ) : (
                      <>Restart the API with the latest code (uvicorn from project root, ideally <code className="inline-code">--reload</code>), then </>
                    )}
                    <button type="button" className="link-inline" onClick={() => void probeHealth()}>
                      Recheck API
                    </button>
                    {!apiHealth.batchScriptMissing && (
                      <>
                        {' '}
                        or use <strong>Reload batch</strong> in the sidebar.
                      </>
                    )}
                  </span>
                )}
              </div>
              <button
                type="button"
                className={
                  'btn btn-primary banner-stale-btn' +
                  (rebuilding ? ' banner-stale-btn--busy' : '') +
                  (!rebuilding && apiHealth.loaded && !apiHealth.rebuildAvailable ? ' banner-stale-btn--blocked' : '')
                }
                disabled={rebuilding || (apiHealth.loaded && !apiHealth.rebuildAvailable)}
                title={
                  apiHealth.loaded && !apiHealth.rebuildAvailable
                    ? 'Update and restart the API first'
                    : undefined
                }
                onClick={startRebuild}
              >
                {rebuilding
                  ? 'Refreshing…'
                  : apiHealth.loaded && !apiHealth.rebuildAvailable
                    ? 'Unavailable'
                    : 'Refresh rankings'}
              </button>
            </div>
          )}
          {dataTab === 'custom' && error && <div className="banner-error">{error}</div>}

          {dataTab === 'custom' && !result && !error && !loading && (
            <p className="hint">Configure symbols and run a scan — results appear in both gap-up and gap-down boards.</p>
          )}

          {showMain && (
            <>
              <div className="stats-strip">
                <div className="stat-card">
                  <div className="label">Universe</div>
                  <div className="value">{dataTab === 'batch' ? batch?.total_symbols ?? '—' : result?.rows.length ?? '—'}</div>
                </div>
                <div className="stat-card">
                  <div className="label">Shown (filters)</div>
                  <div className="value">{dataTab === 'batch' ? filtered.length : result?.rows.length ?? 0}</div>
                </div>
                <div className="stat-card">
                  <div className="label">Threshold</div>
                  <div className="value">
                    {batchParams?.gap_threshold_pct ?? threshold}%
                  </div>
                </div>
                <div className="stat-card">
                  <div className="label">Horizon</div>
                  <div className="value">{batchParams?.forward_trading_days ?? forwardDays} td</div>
                </div>
              </div>

              <GapSection
                variant="up"
                rows={activeRows}
                emptyHint={dataTab === 'batch' ? 'No rows match filters.' : 'No results.'}
              />
              <GapSection
                variant="down"
                rows={activeRows}
                emptyHint={dataTab === 'batch' ? 'No rows match filters.' : 'No results.'}
              />

              {activeErrors.length > 0 && (
                <details className="err-details">
                  <summary>{activeErrors.length} symbols with errors</summary>
                  <ul className="errors">
                    {activeErrors.map((e) => (
                      <li key={e.symbol}>
                        <strong>{e.symbol}</strong>: {e.error}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </>
          )}
        </main>
      </div>

      <footer className="shell-footer">
        Not financial advice. Forecasts and quantile bands are experimental — validate out-of-sample before production use.
      </footer>
    </div>
  )
}

function GapSection({ variant, rows, emptyHint }: { variant: 'up' | 'down'; rows: Row[]; emptyHint: string }) {
  const isUp = variant === 'up'
  const [sort, setSort] = useState(isUp ? DEFAULT_SORT_UP : DEFAULT_SORT_DOWN)

  const sorted = useMemo(() => {
    if (isUp) return sortRowsUp(rows, sort as { key: SortKeyUp; dir: 'asc' | 'desc' })
    return sortRowsDown(rows, sort as { key: SortKeyDown; dir: 'asc' | 'desc' })
  }, [rows, sort, isUp])

  const onSort = useCallback((key: string) => {
    setSort((prev) => {
      if (prev.key === key) return { ...prev, dir: prev.dir === 'asc' ? 'desc' : 'asc' } as typeof prev
      const def = defaultDirForKey(key)
      return { key, dir: def } as typeof prev
    })
  }, [])

  const resetSort = useCallback(() => {
    setSort(isUp ? DEFAULT_SORT_UP : DEFAULT_SORT_DOWN)
  }, [isUp])

  return (
    <section className={'section-card ' + (isUp ? 'up' : 'down')}>
      <div className={'section-head ' + (isUp ? 'up' : 'down')}>
        <div className="section-title">
          <div className={'section-icon ' + (isUp ? 'up' : 'down')}>{isUp ? '↑' : '↓'}</div>
          <div>
            <h2>{isUp ? 'Gap-up board' : 'Gap-down board'}</h2>
            <p>
              {isUp
                ? 'Ranked by implied positive overnight gap tail — use for names with the largest model-suggested upside gaps over your horizon.'
                : 'Ranked by implied negative overnight gap tail — parallel downside view (min forecast & lower quantile). ML column still reflects upside-trained probability.'}
            </p>
          </div>
        </div>
        <div className="section-meta">{sorted.length} rows</div>
      </div>

      {rows.length === 0 ? (
        <p className="hint" style={{ padding: '1rem 1.25rem 1.25rem' }}>
          {emptyHint}
        </p>
      ) : (
        <div className="table-scroll">
          <div className="table-toolbar">
            <button type="button" className="link-reset" onClick={resetSort}>
              Reset sort
            </button>
          </div>
          <table>
            <thead>
              <tr>
                {isUp ? (
                  <>
                    <SortTh label="Symbol" title={TT_UP.symbol} active={sort.key === 'symbol'} dir={sort.dir} onClick={() => onSort('symbol')} />
                    <SortTh label="Upside risk" title={TT_UP.riskScore} active={sort.key === 'risk_score'} dir={sort.dir} onClick={() => onSort('risk_score')} />
                    <SortTh label="Max pt %" title={TT_UP.maxPt} active={sort.key === 'max_point_gap_next_pct'} dir={sort.dir} onClick={() => onSort('max_point_gap_next_pct')} />
                    <SortTh label="Q90 %" title={TT_UP.q90} active={sort.key === 'max_q90_proxy_pct'} dir={sort.dir} onClick={() => onSort('max_q90_proxy_pct')} />
                    <SortTh label="ML P" title={TT_UP.ml} active={sort.key === 'ml_probability'} dir={sort.dir} onClick={() => onSort('ml_probability')} />
                    <SortTh label="Last $" title={TT_UP.lastClose} active={sort.key === 'last_close'} dir={sort.dir} onClick={() => onSort('last_close')} />
                    <SortTh label="Tgt $" title={TT_UP.tgt} active={sort.key === 'implied_target_px'} dir={sort.dir} onClick={() => onSort('implied_target_px')} />
                    <SortTh label="≤ td" title={TT_UP.within} active={sort.key === 'forward_sessions'} dir={sort.dir} onClick={() => onSort('forward_sessions')} />
                    <SortTh label="Scored @" title={TT_UP.scored} active={sort.key === 'ranking_at'} dir={sort.dir} onClick={() => onSort('ranking_at')} />
                    <SortTh label="Bar date" title={TT_UP.lastDate} active={sort.key === 'last_date'} dir={sort.dir} onClick={() => onSort('last_date')} />
                  </>
                ) : (
                  <>
                    <SortTh label="Symbol" title={TT_DN.symbol} active={sort.key === 'symbol'} dir={sort.dir} onClick={() => onSort('symbol')} />
                    <SortTh label="Downside risk" title={TT_DN.riskDown} active={sort.key === 'risk_score_down'} dir={sort.dir} onClick={() => onSort('risk_score_down')} />
                    <SortTh label="Min pt %" title={TT_DN.minPt} active={sort.key === 'min_point_gap_next_pct'} dir={sort.dir} onClick={() => onSort('min_point_gap_next_pct')} />
                    <SortTh label="Q10 %" title={TT_DN.q10} active={sort.key === 'min_q10_proxy_pct'} dir={sort.dir} onClick={() => onSort('min_q10_proxy_pct')} />
                    <SortTh label="ML P" title={TT_DN.ml} active={sort.key === 'ml_probability'} dir={sort.dir} onClick={() => onSort('ml_probability')} />
                    <SortTh label="Last $" title={TT_DN.lastClose} active={sort.key === 'last_close'} dir={sort.dir} onClick={() => onSort('last_close')} />
                    <SortTh label="Tgt $" title={TT_DN.tgt} active={sort.key === 'implied_down_target_px'} dir={sort.dir} onClick={() => onSort('implied_down_target_px')} />
                    <SortTh label="≤ td" title={TT_DN.within} active={sort.key === 'forward_sessions'} dir={sort.dir} onClick={() => onSort('forward_sessions')} />
                    <SortTh label="Scored @" title={TT_DN.scored} active={sort.key === 'ranking_at'} dir={sort.dir} onClick={() => onSort('ranking_at')} />
                    <SortTh label="Bar date" title={TT_DN.lastDate} active={sort.key === 'last_date'} dir={sort.dir} onClick={() => onSort('last_date')} />
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr
                  key={r.symbol + variant}
                  className={
                    isUp
                      ? r.risk_score >= 50
                        ? 'heat-up-high'
                        : r.risk_score >= 25
                          ? 'heat-up-mid'
                          : ''
                      : (r.risk_score_down ?? 0) >= 50
                        ? 'heat-dn-high'
                        : (r.risk_score_down ?? 0) >= 25
                          ? 'heat-dn-mid'
                          : ''
                  }
                >
                  {isUp ? (
                    <>
                      <td className="sym">{r.symbol}</td>
                      <td className="num">{r.risk_score.toFixed(2)}</td>
                      <td className="num">{r.max_point_gap_next_pct.toFixed(2)}</td>
                      <td className="num">{r.max_q90_proxy_pct.toFixed(2)}</td>
                      <td className="num">{fmtMl(r)}</td>
                      <td className="num">{fmtPx(r.last_close)}</td>
                      <td className="num">{fmtPx(r.implied_target_px)}</td>
                      <td className="num">{fmtSessions(r.forward_sessions)}</td>
                      <td className="dim-cell">{fmtScored(r.ranking_at)}</td>
                      <td className="num">{r.last_date}</td>
                    </>
                  ) : (
                    <>
                      <td className="sym">{r.symbol}</td>
                      <td className="num">{r.risk_score_down != null ? Number(r.risk_score_down).toFixed(2) : '—'}</td>
                      <td className="num">{r.min_point_gap_next_pct != null ? Number(r.min_point_gap_next_pct).toFixed(2) : '—'}</td>
                      <td className="num">{r.min_q10_proxy_pct != null ? Number(r.min_q10_proxy_pct).toFixed(2) : '—'}</td>
                      <td className="num">{fmtMl(r)}</td>
                      <td className="num">{fmtPx(r.last_close)}</td>
                      <td className="num">{fmtPx(r.implied_down_target_px)}</td>
                      <td className="num">{fmtSessions(r.forward_sessions)}</td>
                      <td className="dim-cell">{fmtScored(r.ranking_at)}</td>
                      <td className="num">{r.last_date}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function defaultDirForKey(key: string): 'asc' | 'desc' {
  if (key === 'symbol') return 'asc'
  return 'desc'
}

function parseRankingTime(s: string | undefined | null): number {
  if (!s) return 0
  const t = Date.parse(s)
  return Number.isNaN(t) ? 0 : t
}

function hasMlProb(r: Row): boolean {
  return r.ml_probability != null && r.ml_probability !== undefined && !Number.isNaN(r.ml_probability)
}

function sortRowsUp(rows: Row[], sort: { key: SortKeyUp; dir: 'asc' | 'desc' }): Row[] {
  const sign = sort.dir === 'asc' ? 1 : -1
  const out = [...rows]
  out.sort((a, b) => {
    const c = (() => {
      switch (sort.key) {
        case 'symbol':
          return sign * a.symbol.localeCompare(b.symbol)
        case 'risk_score':
          return sign * (a.risk_score - b.risk_score)
        case 'max_point_gap_next_pct':
          return sign * (a.max_point_gap_next_pct - b.max_point_gap_next_pct)
        case 'max_q90_proxy_pct':
          return sign * (a.max_q90_proxy_pct - b.max_q90_proxy_pct)
        case 'last_date':
          return sign * (new Date(a.last_date).getTime() - new Date(b.last_date).getTime())
        case 'last_close':
          return cmpOptNum(a.last_close, b.last_close, sign)
        case 'implied_target_px':
          return cmpOptNum(a.implied_target_px, b.implied_target_px, sign)
        case 'forward_sessions':
          return cmpOptNum(a.forward_sessions, b.forward_sessions, sign)
        case 'ranking_at':
          return cmpOptTime(a.ranking_at, b.ranking_at, sign)
        case 'ml_probability':
          return cmpMl(a, b, sign)
        default:
          return 0
      }
    })()
    if (c !== 0) return c
    return a.symbol.localeCompare(b.symbol)
  })
  return out
}

function sortRowsDown(rows: Row[], sort: { key: SortKeyDown; dir: 'asc' | 'desc' }): Row[] {
  const sign = sort.dir === 'asc' ? 1 : -1
  const out = [...rows]
  out.sort((a, b) => {
    const c = (() => {
      switch (sort.key) {
        case 'symbol':
          return sign * a.symbol.localeCompare(b.symbol)
        case 'risk_score_down':
          return cmpOptNum(a.risk_score_down ?? null, b.risk_score_down ?? null, sign)
        case 'min_point_gap_next_pct':
          return cmpOptNum(a.min_point_gap_next_pct ?? null, b.min_point_gap_next_pct ?? null, sign)
        case 'min_q10_proxy_pct':
          return cmpOptNum(a.min_q10_proxy_pct ?? null, b.min_q10_proxy_pct ?? null, sign)
        case 'last_date':
          return sign * (new Date(a.last_date).getTime() - new Date(b.last_date).getTime())
        case 'last_close':
          return cmpOptNum(a.last_close, b.last_close, sign)
        case 'implied_down_target_px':
          return cmpOptNum(a.implied_down_target_px ?? null, b.implied_down_target_px ?? null, sign)
        case 'forward_sessions':
          return cmpOptNum(a.forward_sessions, b.forward_sessions, sign)
        case 'ranking_at':
          return cmpOptTime(a.ranking_at, b.ranking_at, sign)
        case 'ml_probability':
          return cmpMl(a, b, sign)
        default:
          return 0
      }
    })()
    if (c !== 0) return c
    return a.symbol.localeCompare(b.symbol)
  })
  return out
}

function cmpOptNum(a: number | null | undefined, b: number | null | undefined, sign: number): number {
  const ha = a != null && !Number.isNaN(a)
  const hb = b != null && !Number.isNaN(b)
  if (!ha && !hb) return 0
  if (!ha) return 1
  if (!hb) return -1
  return sign * (a! - b!)
}

function cmpOptTime(a: string | null | undefined, b: string | null | undefined, sign: number): number {
  const ha = !!a
  const hb = !!b
  if (!ha && !hb) return 0
  if (!ha) return 1
  if (!hb) return -1
  return sign * (parseRankingTime(a) - parseRankingTime(b))
}

function cmpMl(a: Row, b: Row, sign: number): number {
  const ha = hasMlProb(a)
  const hb = hasMlProb(b)
  if (!ha && !hb) return 0
  if (!ha) return 1
  if (!hb) return -1
  return sign * ((a.ml_probability as number) - (b.ml_probability as number))
}

function fmtPx(n: number | null | undefined) {
  if (n == null || Number.isNaN(n)) return '—'
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function fmtSessions(n: number | null | undefined) {
  if (n == null || Number.isNaN(n)) return '—'
  return `≤${n} td`
}

function fmtScored(iso: string | null | undefined) {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
}

function fmtMl(r: Row) {
  if (r.ml_probability != null && r.ml_probability !== undefined) return r.ml_probability.toFixed(4)
  if (r.ml_skip_reason === 'no_model_files') return '—'
  return r.ml_skip_reason ?? '—'
}

function SortTh({
  label,
  title,
  active,
  dir,
  onClick,
}: {
  label: string
  title: string
  active: boolean
  dir: 'asc' | 'desc'
  onClick: () => void
}) {
  return (
    <th className="th-sort-cell">
      <button
        type="button"
        className={'th-sort' + (active ? ' active' : '')}
        title={title}
        onClick={onClick}
        aria-sort={active ? (dir === 'asc' ? 'ascending' : 'descending') : 'none'}
      >
        <span>{label}</span>
        <span className="caret" aria-hidden>
          {active ? (dir === 'asc' ? '▲' : '▼') : ''}
        </span>
      </button>
    </th>
  )
}
