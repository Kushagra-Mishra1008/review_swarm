    /**
 * EvalDashboard — the plan's "second screen": recall/precision from
 * Phase 5, swarm vs single-agent baseline side by side, tokens per
 * review. Static, data-dense, no animation — deliberately different
 * register from the live review screen.
 */

import { useEffect, useState } from 'react';

function fmtPct(v) {
  return v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`;
}

function fmtTokens(v) {
  return v === null || v === undefined ? '—' : Math.round(v).toLocaleString();
}

function MetricRow({ label, swarmValue, baselineValue, formatter }) {
  return (
    <div className="metric-row">
      <span className="metric-row__label">{label}</span>
      <span className="metric-row__value">{formatter(swarmValue)}</span>
      <span className="metric-row__value metric-row__value--baseline">{formatter(baselineValue)}</span>
    </div>
  );
}

export default function EvalDashboard({ apiBase }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${apiBase}/eval`)
      .then((res) => {
        if (!res.ok) throw new Error(res.status === 404 ? 'No eval report yet — run eval/report.py first.' : `Server returned ${res.status}`);
        return res.json();
      })
      .then(setReport)
      .catch((err) => setError(err.message));
  }, [apiBase]);

  if (error) {
    return (
      <main className="eval-dashboard eval-dashboard--empty">
        <p>{error}</p>
      </main>
    );
  }

  if (!report) {
    return (
      <main className="eval-dashboard eval-dashboard--empty">
        <p>Loading evaluation report…</p>
      </main>
    );
  }

  const { summary, pr_count, novel_findings_for_manual_review, per_pr } = report;

  return (
    <main className="eval-dashboard">
      <section className="eval-summary">
        <div className="eval-summary__title">
          EVALUATION — {pr_count} PRs scored
        </div>

        <div className="metric-table">
          <div className="metric-row metric-row--header">
            <span className="metric-row__label"></span>
            <span className="metric-row__value">SWARM</span>
            <span className="metric-row__value metric-row__value--baseline">BASELINE</span>
          </div>
          <MetricRow
            label="Recall"
            swarmValue={summary.swarm.mean_recall}
            baselineValue={summary.baseline.mean_recall}
            formatter={fmtPct}
          />
          <MetricRow
            label="Precision"
            swarmValue={summary.swarm.mean_precision}
            baselineValue={summary.baseline.mean_precision}
            formatter={fmtPct}
          />
          <MetricRow
            label="Tokens / review"
            swarmValue={summary.swarm.mean_tokens_per_review}
            baselineValue={summary.baseline.mean_tokens_per_review}
            formatter={fmtTokens}
          />
        </div>
      </section>

      <section className="eval-novel">
        <div className="eval-novel__title">
          NOVEL FINDINGS ({novel_findings_for_manual_review?.length || 0}) — caught by the swarm, not flagged by human reviewers
        </div>
        <div className="eval-novel__list">
          {(novel_findings_for_manual_review || []).slice(0, 20).map((f, i) => (
            <div key={i} className="eval-novel__item">
              <span className="eval-novel__pr">#{f.pr_number}</span>
              <span className={`eval-novel__severity eval-novel__severity--${f.severity}`}>
                {f.severity?.toUpperCase()}
              </span>
              <span className="eval-novel__msg">{f.message}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="eval-per-pr">
        <div className="eval-per-pr__title">PER-PR BREAKDOWN</div>
        <div className="eval-per-pr__table">
          <div className="eval-per-pr__row eval-per-pr__row--header">
            <span>PR</span>
            <span>Recall</span>
            <span>Precision</span>
            <span>Tokens</span>
            <span>Human comments</span>
          </div>
          {(per_pr || []).map((r) => (
            <div key={r.pr_number} className="eval-per-pr__row">
              <a href={r.pr_url} target="_blank" rel="noreferrer">#{r.pr_number}</a>
              <span>{fmtPct(r.swarm_recall)}</span>
              <span>{fmtPct(r.swarm_precision)}</span>
              <span>{fmtTokens(r.swarm_tokens)}</span>
              <span>{r.human_comment_count}</span>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}