import { useState, useEffect } from 'react';
import { useEventStream } from './hooks/useEventStream';
import { useStaticReplay } from './hooks/useStaticReplay';
import AgentGraph from './components/AgentGraph';
import EventStream from './components/EventStream';
import FindingsPanel from './components/FindingsPanel';
import BudgetBar from './components/BudgetBar';
import EvalDashboard from './components/EvalDashboard';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const REPO_URL = 'https://github.com/Kushagra-Mishra1008/review_swarm';

// Baked-in demo run, shipped as a static asset at /recordings/demo.jsonl.
// A visitor opening the deployed link with no query string gets a full
// recorded run with no backend, no tokens, and no network calls.
const DEMO_RUN_ID = 'demo';

// True when there is no backend to talk to - the deployed static build.
// Detected by probing /budget once on load rather than by build flag, so
// the same bundle works locally with the server up or down.
function useBackendAvailable() {
  const [available, setAvailable] = useState(null);
  const [budget, setBudget] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/budget`)
      .then((res) => {
        if (!res.ok) throw new Error('bad status');
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        setBudget(data);
        setAvailable(true);
      })
      .catch(() => {
        if (!cancelled) setAvailable(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { available, budget };
}

function readReplayParam() {
  const params = new URLSearchParams(window.location.search);
  return params.get('replay');
}

export default function App() {
  const [prUrl, setPrUrl] = useState('');
  const [runId, setRunId] = useState(null);
  const [isReplay, setIsReplay] = useState(false);
  const [screen, setScreen] = useState('review');
  const [launchError, setLaunchError] = useState('');

  const { available: backendAvailable, budget: initialBudget } = useBackendAvailable();

  // Static replay reads a .jsonl from the frontend's own public/ dir and
  // paces the events by their recorded timestamps. Used when there is no
  // backend, so the deployed demo never depends on a server being up.
  const staticMode = backendAvailable === false;

  useEffect(() => {
    if (backendAvailable === null) return;

    const fromUrl = readReplayParam();
    if (fromUrl) {
      setIsReplay(true);
      setRunId(fromUrl);
      return;
    }

    // No explicit run requested and no backend to start one - fall back
    // to the bundled demo so the page is never blank.
    if (backendAvailable === false) {
      setIsReplay(true);
      setRunId(DEMO_RUN_ID);
    }
  }, [backendAvailable]);

  // The eval dashboard reads live numbers from the backend, which the
  // static demo build doesn't have. Force the review screen there rather
  // than leaving a tab that resolves to an empty panel.
  useEffect(() => {
    if (staticMode && screen !== 'review') setScreen('review');
  }, [staticMode, screen]);

  const live = useEventStream(staticMode ? null : runId, {
    replay: isReplay,
    apiBase: API_BASE,
    initialBudget,
  });

  const staticReplay = useStaticReplay(staticMode ? runId : null);

  const {
    agentState,
    agentDetail,
    workerCounts,
    events,
    findings,
    budget,
    status,
    reportText,
  } = staticMode ? staticReplay : live;

  async function startReview(e) {
    e.preventDefault();
    setLaunchError('');
    setIsReplay(false);
    setRunId(null);

    try {
      const res = await fetch(`${API_BASE}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pr_url: prUrl }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setRunId(data.run_id);
    } catch (err) {
      setLaunchError(`Could not start review: ${err.message}`);
    }
  }

  function startReplay(recordedRunId) {
    setIsReplay(true);
    setRunId(recordedRunId);
  }

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-title">REVIEW SWARM</span>
        {staticMode && <span className="app-badge">DEMO - RECORDED RUN</span>}
        {staticMode ? (
          <nav className="app-nav">
            <a
              className="nav-btn nav-btn--link"
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
            >
              Source on GitHub
            </a>
          </nav>
        ) : (
          <nav className="app-nav">
            <button
              className={screen === 'review' ? 'nav-btn nav-btn--active' : 'nav-btn'}
              onClick={() => setScreen('review')}
            >
              Review
            </button>
            <button
              className={screen === 'eval' ? 'nav-btn nav-btn--active' : 'nav-btn'}
              onClick={() => setScreen('eval')}
            >
              Evaluation
            </button>
          </nav>
        )}
      </header>

      {screen === 'review' ? (
        <>
          {!staticMode && (
            <form className="pr-form" onSubmit={startReview}>
              <input
                className="pr-input"
                type="text"
                placeholder="https://github.com/owner/repo/pull/123"
                value={prUrl}
                onChange={(e) => setPrUrl(e.target.value)}
                required
              />
              <button className="pr-submit" type="submit" disabled={status === 'streaming'}>
                {status === 'streaming' ? 'Running...' : 'Run review'}
              </button>
              <button
                className="pr-replay"
                type="button"
                onClick={() => {
                  const id = window.prompt('Enter a recorded run_id to replay:');
                  if (id) startReplay(id);
                }}
              >
                Replay
              </button>
            </form>
          )}

          {staticMode && (
            <div className="replay-note">
              Replaying a recorded review of a scikit-learn pull request at original
              timing. No API calls, no tokens spent.
            </div>
          )}

          {launchError && <div className="launch-error">{launchError}</div>}

          <main className="columns">
            <section className="col col--graph">
              <AgentGraph
                agentState={agentState}
                agentDetail={agentDetail}
                workerCounts={workerCounts}
              />
            </section>
            <section className="col col--stream">
              <EventStream events={events} />
            </section>
            <section className="col col--findings">
              <FindingsPanel findings={findings} reportText={reportText} status={status} />
            </section>
          </main>

          <footer className="footer">
            <BudgetBar budget={budget} />
          </footer>
        </>
      ) : (
        <EvalDashboard apiBase={API_BASE} />
      )}
    </div>
  );
}