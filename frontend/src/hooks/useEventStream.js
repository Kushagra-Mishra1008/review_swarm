/**
 * useEventStream — subscribes to a review run's SSE event stream and
 * exposes it as structured React state.
 *
 * agentDetail is what powers the expanded agent cards: everything the
 * run actually revealed about each agent's decisions — which
 * specialists the lead activated, which files each specialist
 * dispatched workers to, and what each worker found. All of it comes
 * from events the backend already emits; nothing here is inferred or
 * invented.
 */

import { useEffect, useRef, useState, useCallback } from 'react';

const PIPELINE_NODES = [
  'fetch_pr',
  'clone_for_static_analysis',
  'connect_repo_index_mcp',
  'run_static_analysis',
  'report',
];

const SPECIALIST_NODES = ['security', 'performance', 'testing', 'maintainability'];

const AGENT_NODES = [...PIPELINE_NODES, 'lead_triage', ...SPECIALIST_NODES];

const initialAgentState = () =>
  Object.fromEntries(AGENT_NODES.map((n) => [n, 'idle']));

const initialAgentDetail = () =>
  Object.fromEntries(AGENT_NODES.map((n) => [n, { workers: [], findingCount: null, meta: {} }]));

const initialWorkerCounts = () =>
  Object.fromEntries(SPECIALIST_NODES.map((n) => [n, 0]));

export function useEventStream(
  runId,
  { replay = false, apiBase = 'http://localhost:8000', initialBudget = null } = {}
) {
  const [events, setEvents] = useState([]);
  const [agentState, setAgentState] = useState(initialAgentState);
  const [agentDetail, setAgentDetail] = useState(initialAgentDetail);
  const [workerCounts, setWorkerCounts] = useState(initialWorkerCounts);
  const [findings, setFindings] = useState([]);
  const [budget, setBudget] = useState({ tpdSpent: 0, tpdLimit: 200000, throttled: false });
  const [status, setStatus] = useState('idle');
  const [reportText, setReportText] = useState('');
  const eventSourceRef = useRef(null);

  useEffect(() => {
    if (initialBudget) {
      setBudget((prev) => ({
        ...prev,
        tpdSpent: initialBudget.tpd_spent ?? prev.tpdSpent,
        tpdLimit: initialBudget.tpd_limit ?? prev.tpdLimit,
      }));
    }
  }, [initialBudget]);

  const reset = useCallback(() => {
    setEvents([]);
    setAgentState(initialAgentState());
    setAgentDetail(initialAgentDetail());
    setWorkerCounts(initialWorkerCounts());
    setFindings([]);
    setReportText('');
    setBudget((prev) => ({ ...prev, throttled: false }));
    // tpdSpent is deliberately preserved — the daily budget doesn't
    // reset between runs, so neither should the bar.
  }, []);

  useEffect(() => {
    if (!runId) return;

    reset();
    setStatus('connecting');

    const url = `${apiBase}/stream/${runId}${replay ? '?replay=true' : ''}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onopen = () => setStatus('streaming');

    es.onmessage = (msg) => {
      const event = JSON.parse(msg.data);
      setEvents((prev) => [...prev, event]);

      const { event_type, data } = event;

      if (event_type === 'node_start' && AGENT_NODES.includes(data.node)) {
        setAgentState((prev) => ({ ...prev, [data.node]: 'active' }));
      }

      else if (event_type === 'node_complete' && AGENT_NODES.includes(data.node)) {
        setAgentState((prev) => ({ ...prev, [data.node]: 'done' }));
        setAgentDetail((prev) => ({
          ...prev,
          [data.node]: {
            ...prev[data.node],
            findingCount: data.finding_count ?? prev[data.node].findingCount,
            meta: { ...prev[data.node].meta, ...data },
          },
        }));

        // The lead's decision determines which specialists never run at
        // all — an unactivated specialist is 'skipped', not 'idle'.
        if (data.node === 'lead_triage' && Array.isArray(data.active_specialists)) {
          setAgentState((prev) => {
            const next = { ...prev };
            for (const s of SPECIALIST_NODES) {
              if (!data.active_specialists.includes(s)) next[s] = 'skipped';
            }
            return next;
          });
        }
      }

      else if (event_type === 'throttle_wait') {
        setBudget((prev) => ({ ...prev, throttled: true }));
        setAgentState((prev) => {
          const next = { ...prev };
          for (const key of Object.keys(next)) {
            if (next[key] === 'active') next[key] = 'waiting';
          }
          return next;
        });
      }

      else if (event_type === 'worker_spawn' && data.specialist) {
        setWorkerCounts((prev) => ({
          ...prev,
          [data.specialist]: (prev[data.specialist] || 0) + 1,
        }));
        setAgentDetail((prev) => ({
          ...prev,
          [data.specialist]: {
            ...prev[data.specialist],
            workers: [
              ...prev[data.specialist].workers,
              { file: data.file, status: 'running', findingCount: null },
            ],
          },
        }));
      }

      else if (event_type === 'worker_complete' && data.specialist) {
        setWorkerCounts((prev) => ({
          ...prev,
          [data.specialist]: Math.max((prev[data.specialist] || 0) - 1, 0),
        }));
        setAgentDetail((prev) => ({
          ...prev,
          [data.specialist]: {
            ...prev[data.specialist],
            workers: prev[data.specialist].workers.map((w) =>
              w.file === data.file && w.status === 'running'
                ? { ...w, status: 'done', findingCount: data.finding_count }
                : w
            ),
          },
        }));
      }

      else if (event_type === 'finding') {
        setFindings((prev) => [...prev, data]);
      }

      else if (event_type === 'cache_hit' || event_type === 'llm_call_complete') {
        setBudget((prev) => ({
          ...prev,
          tpdSpent: data.tpd_spent ?? prev.tpdSpent,
          throttled: false,
        }));
      }

      else if (event_type === 'run_complete') {
        setReportText(data.report_text || '');
        setStatus('complete');
        setBudget((prev) => ({ ...prev, throttled: false }));
        es.close();
      }
    };

    es.onerror = () => {
      setStatus('error');
      es.close();
    };

    return () => {
      es.close();
    };
  }, [runId, replay, apiBase, reset]);

  return { events, agentState, agentDetail, workerCounts, findings, budget, status, reportText };
}