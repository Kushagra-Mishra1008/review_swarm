/**
 * FindingsPanel — findings grouped by severity, click to expand into
 * the agent's full message and suggested fix. Shows the final report
 * summary once the run completes.
 */

import { useState } from 'react';

const SEVERITY_ORDER = ['blocker', 'major', 'minor', 'nit'];

function groupBySeverity(findings) {
  const groups = Object.fromEntries(SEVERITY_ORDER.map((s) => [s, []]));
  for (const f of findings) {
    const key = SEVERITY_ORDER.includes(f.severity) ? f.severity : 'nit';
    groups[key].push(f);
  }
  return groups;
}

function FindingRow({ finding }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`finding finding--${finding.severity}`}>
      <button className="finding__summary" onClick={() => setExpanded((e) => !e)}>
        <span className="finding__marker">▲</span>
        <span className="finding__location">
          {finding.file}:{finding.line}
        </span>
        <span className="finding__category">{finding.category}</span>
      </button>
      {expanded && (
        <div className="finding__detail">
          <p className="finding__message">{finding.message}</p>
          {finding.suggested_fix && (
            <p className="finding__fix">
              <span className="finding__fix-label">fix:</span> {finding.suggested_fix}
            </p>
          )}
          <p className="finding__agent">via {finding.source_agent}</p>
        </div>
      )}
    </div>
  );
}

export default function FindingsPanel({ findings, reportText, status }) {
  const grouped = groupBySeverity(findings);

  return (
    <div className="panel findings-panel">
      <div className="panel__header">
        FINDINGS
        {findings.length > 0 && <span className="panel__count">{findings.length}</span>}
      </div>
      <div className="findings-panel__body">
        {findings.length === 0 && status !== 'complete' && (
          <div className="findings-panel__empty">No findings yet…</div>
        )}
        {findings.length === 0 && status === 'complete' && (
          <div className="findings-panel__empty">Clean — no issues found.</div>
        )}
        {SEVERITY_ORDER.map(
          (severity) =>
            grouped[severity].length > 0 && (
              <div key={severity} className="findings-panel__group">
                <div className={`findings-panel__group-label findings-panel__group-label--${severity}`}>
                  {severity.toUpperCase()} ({grouped[severity].length})
                </div>
                {grouped[severity].map((f, i) => (
                  <FindingRow key={i} finding={f} />
                ))}
              </div>
            )
        )}
        {status === 'complete' && reportText && (
          <pre className="findings-panel__report">{reportText}</pre>
        )}
      </div>
    </div>
  );
}