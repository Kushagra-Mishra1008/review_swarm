/**
 * useStaticReplay — plays a recorded run from a static .jsonl asset,
 * with no backend and no network beyond one file fetch.
 *
 * The deployed demo has no server: Vercel serves the bundle and nothing
 * else. This hook reads backend/data/recordings/<id>.jsonl (copied into
 * frontend/public/recordings/ at build time), paces the events by their
 * recorded timestamps, and exposes exactly the same shape useEventStream
 * returns so App.jsx can swap between them without the components
 * knowing which is which.
 *
 * Playback is capped so a run that sat waiting on the token bucket for
 * 40 seconds doesn't make a demo viewer stare at a frozen screen.
 */

import { useEffect, useRef, useState } from 'react';

const PIPELINE_NODES = [
  'fetch_pr',
  'clone_for_static_analysis',
  'connect_repo_index_mcp',
  'run_static_analysis',
  'report',
];

const SPECIALIST_NODES = ['security', 'performance', 'testing', 'maintainability'];

const AGENT_NODES = [...PIPELINE_NODES, 'lead_triage', ...SPECIALIST_NODES];

// Real runs include multi-second throttle waits. Preserving them exactly
// makes for a dull demo, so gaps are compressed past this ceiling.
const MAX_GAP_MS = 1500;

const initialAgentState = () =>
  Object.fromEntries(AGENT_NODES.map((n) => [n, 'idle']));

const initialAgentDetail = () =>
  Object.fromEntries(
    AGENT_NODES.map((n) => [n, { workers: [], findingCount: null, meta: {} }])
  );

const initialWorkerCounts = () =>
  Object.fromEntries(SPECIALIST_NODES.map((n) => [n, 0]));

export function useStaticReplay(runId) {
  const [events, setEvents] = useState([]);
  const [agentState, setAgentState] = useState(initialAgentState);
  const [agentDetail, setAgentDetail] = useState(initialAgentDetail);
  const [workerCounts, setWorkerCounts] = useState(initialWorkerCounts);
  const [findings, setFindings] = useState([]);
  const [budget, setBudget] = useState({ tpdSpent: 0, tpdLimit: 200000, throttled: false });
  const [status, setStatus] = useState('idle');
  const [reportText, setReportText] = useState('');
  const timersRef = useRef([]);

  useEffect(() => {
    if (!runId) return;

    let cancelled = false;

    // Clear any playback already in flight.
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];

    setEvents([]);
    setAgentState(initialAgentState());
    setAgentDetail(initialAgentDetail());
    setWorkerCounts(initialWorkerCounts());
    setFindings([]);
    setReportText('');
    setBudget({ tpdSpent: 0, tpdLimit: 200000, throttled: false });
    setStatus('connecting');

    fetch(`${import.meta.env.BASE_URL}recordings/${runId}.jsonl`)
      .then((res) => {
        if (!res.ok) throw new Error(`recording not found (${res.status})`);
        return res.text();
      })
      .then((text) => {
        if (cancelled) return;

        const parsed = text
          .split('\n')
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => {
            try {
              return JSON.parse(line);
            } catch {
              return null;
            }
          })
          .filter(Boolean);

        if (parsed.length === 0) {
          setStatus('error');
          return;
        }

        setStatus('streaming');

        // Convert absolute timestamps into a capped cumulative schedule.
        const base = parsed[0].timestamp ?? 0;
        let elapsed = 0;
        let previous = base;

        parsed.forEach((event) => {
          const ts = event.timestamp ?? previous;
          const gap = Math.max(0, (ts - previous) * 1000);
          elapsed += Math.min(gap, MAX_GAP_MS);
          previous = ts;

          const timer = setTimeout(() => {
            if (!cancelled) applyEvent(event);
          }, elapsed);
          timersRef.current.push(timer);
        });
      })
      .catch(() => {
        if (!cancelled) setStatus('error');
      });

    function applyEvent(event) {
      setEvents((prev) => [...prev, event]);

      const { event_type, data = {} } = event;

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
          tpdLimit: data.tpd_limit ?? prev.tpdLimit,
          throttled: false,
        }));
      }

      else if (event_type === 'run_complete') {
        setReportText(data.report_text || '');
        setStatus('complete');
        setBudget((prev) => ({ ...prev, throttled: false }));
      }
    }

    return () => {
      cancelled = true;
      timersRef.current.forEach(clearTimeout);
      timersRef.current = [];
    };
  }, [runId]);

  return { events, agentState, agentDetail, workerCounts, findings, budget, status, reportText };
}