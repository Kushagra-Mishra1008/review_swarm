/**
 * AgentGraph — the pipeline as interactive cards. Click any card to
 * expand it and see what that agent does, what it looks for, what it
 * decided this run, and which files its workers scanned.
 *
 * Everything in the expanded state is either static role description
 * or data the run actually emitted. Per-agent token cost is
 * deliberately absent: the gateway reports token spend globally, not
 * attributed to a calling agent, so showing it here would be a guess.
 */

import { useState } from 'react';

const PIPELINE_HEAD = [
  {
    key: 'fetch_pr',
    label: 'fetch_pr',
    tier: 'pipeline',
    model: null,
    does: 'Pulls the pull request — metadata, changed files, and diff — through the official GitHub MCP server.',
  },
  {
    key: 'clone_for_static_analysis',
    label: 'clone',
    tier: 'pipeline',
    model: null,
    does: 'Shallow-clones the PR branch to a temp directory so linters and code search have real files to work with.',
  },
  {
    key: 'connect_repo_index_mcp',
    label: 'repo_index_mcp',
    tier: 'pipeline',
    model: null,
    does: 'Starts the self-authored Repo Index MCP server against the clone, exposing semantic code search over the repo.',
  },
  {
    key: 'run_static_analysis',
    label: 'static_analysis',
    tier: 'pipeline',
    model: null,
    does: 'Runs ruff and semgrep. Their findings are handed to specialists as evidence to verify — cheap detection, LLM judgment.',
  },
];

const LEAD = {
  key: 'lead_triage',
  label: 'lead',
  tier: 'orchestrator',
  model: 'large',
  does: 'Reads a preview of the changed code and decides which specialists to activate. Never reviews code itself.',
};

const SPECIALISTS = [
  {
    key: 'security',
    label: 'security',
    tier: 'specialist',
    model: 'large',
    does: 'Selects security-relevant files, then dispatches a worker to scan each one.',
    looksFor: [
      'hardcoded secrets and credentials',
      'SQL and command injection',
      'unsafe deserialization',
      'missing auth checks',
    ],
  },
  {
    key: 'performance',
    label: 'performance',
    tier: 'specialist',
    model: 'large',
    does: 'Selects files with runtime-relevant changes, then dispatches a worker to scan each one.',
    looksFor: [
      'N+1 query patterns',
      'expensive calls inside loops',
      'blocking I/O in hot paths',
      'unbounded queries',
    ],
  },
  {
    key: 'testing',
    label: 'testing',
    tier: 'specialist',
    model: 'large',
    does: 'Selects files with new logic or weak tests, then dispatches a worker to scan each one.',
    looksFor: [
      'untested new logic',
      'assertions that verify nothing',
      'unhandled edge cases',
      'untested error paths',
    ],
  },
  {
    key: 'maintainability',
    label: 'maintainability',
    tier: 'specialist',
    model: 'large',
    does: 'Selects files with logic worth assessing, then dispatches a worker to scan each one.',
    looksFor: [
      'functions doing too many things',
      'duplicated logic',
      'magic numbers and strings',
      'unclear naming',
    ],
  },
];

const REPORT = {
  key: 'report',
  label: 'report',
  tier: 'pipeline',
  model: null,
  does: 'Dedupes findings across specialists, ranks by severity, and produces the final summary. Pure Python — no tokens.',
};

const STATUS_LABEL = {
  idle: 'IDLE',
  active: 'ACTIVE',
  waiting: 'WAITING',
  done: 'DONE',
  skipped: 'SKIPPED',
};

function StatusMarker({ state }) {
  if (state === 'done') return <span className="card__marker card__marker--done">✓</span>;
  if (state === 'skipped') return <span className="card__marker card__marker--skipped">⊘</span>;
  if (state === 'idle') return <span className="card__marker card__marker--idle">○</span>;
  return <span className="card__marker card__marker--pulse" />;
}

function decisionFor(agent, detail) {
  const meta = detail?.meta || {};

  if (agent.key === 'lead_triage' && Array.isArray(meta.active_specialists)) {
    return meta.active_specialists.length
      ? `Activated: ${meta.active_specialists.join(', ')}`
      : 'Activated no specialists.';
  }
  if (agent.key === 'fetch_pr' && meta.file_count != null) {
    return `${meta.file_count} changed files in this PR.`;
  }
  if (agent.key === 'connect_repo_index_mcp' && Array.isArray(meta.tools)) {
    return `Tools exposed: ${meta.tools.join(', ')}`;
  }
  if (agent.key === 'connect_repo_index_mcp' && meta.skipped) {
    return 'Skipped — no local clone was available.';
  }
  if (agent.key === 'run_static_analysis' && meta.files_with_evidence != null) {
    return `${meta.files_with_evidence} files had machine-detected issues to verify.`;
  }
  if (agent.key === 'report' && meta.final_finding_count != null) {
    return `${meta.final_finding_count} findings after dedupe and ranking.`;
  }
  return null;
}

function AgentCard({ agent, state, detail, workerCount, indented }) {
  const [open, setOpen] = useState(false);
  const workers = detail?.workers || [];
  const decision = decisionFor(agent, detail);
  const isSpecialist = agent.tier === 'specialist';

  return (
    <div className={`card card--${state} ${indented ? 'card--indented' : ''}`}>
      <button
        className="card__head"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <div className="card__identity">
          <div className="card__name-row">
            <StatusMarker state={state} />
            <span className="card__name">{agent.label}</span>
          </div>
          <div className="card__meta">
            {agent.tier}
            {agent.model && ` · ${agent.model} model`}
          </div>
        </div>

        <div className="card__status-col">
          <span className={`card__badge card__badge--${state}`}>{STATUS_LABEL[state]}</span>
          {workerCount > 0 && <span className="card__workers">×{workerCount} workers</span>}
          {state === 'done' && detail?.findingCount != null && (
            <span className="card__workers">{detail.findingCount} findings</span>
          )}
        </div>
      </button>

      {open && (
        <div className="card__body">
          <p className="card__does">{agent.does}</p>

          {agent.looksFor && (
            <>
              <div className="card__section-label">LOOKS FOR</div>
              <ul className="card__list">
                {agent.looksFor.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          )}

          {decision && (
            <>
              <div className="card__section-label">THIS RUN</div>
              <p className="card__decision">{decision}</p>
            </>
          )}

          {isSpecialist && (
            <>
              <div className="card__section-label">
                WORKERS DISPATCHED {workers.length > 0 && `(${workers.length})`}
              </div>
              {workers.length === 0 ? (
                <p className="card__decision card__decision--muted">
                  {state === 'skipped'
                    ? 'Not activated for this PR.'
                    : state === 'done'
                    ? 'No files selected for scanning.'
                    : 'None yet.'}
                </p>
              ) : (
                <ul className="card__workers-list">
                  {workers.map((w, i) => (
                    <li key={`${w.file}-${i}`} className={`worker worker--${w.status}`}>
                      <span className="worker__file">{w.file}</span>
                      <span className="worker__result">
                        {w.status === 'running'
                          ? 'scanning…'
                          : `${w.findingCount} finding${w.findingCount === 1 ? '' : 's'}`}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function AgentGraph({ agentState = {}, agentDetail = {}, workerCounts = {} }) {
  const render = (agent, indented = false) => (
    <AgentCard
      key={agent.key}
      agent={agent}
      state={agentState[agent.key] || 'idle'}
      detail={agentDetail[agent.key]}
      workerCount={workerCounts[agent.key] || 0}
      indented={indented}
    />
  );

  return (
    <div className="panel agent-graph">
      <div className="panel__header">AGENT GRAPH</div>
      <div className="graph-body">
        {PIPELINE_HEAD.map((a) => render(a))}

        <div className="graph-rule" />
        {render(LEAD)}

        <div className="graph-branch">├─ specialists · parallel</div>
        <div className="graph-branch-group">{SPECIALISTS.map((a) => render(a, true))}</div>

        <div className="graph-rule" />
        {render(REPORT)}
      </div>
    </div>
  );
}