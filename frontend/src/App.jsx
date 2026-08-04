import { useState, useEffect } from 'react';
import { useEventStream } from './hooks/useEventStream';
import AgentGraph from './components/AgentGraph';
import EventStream from './components/EventStream';
import FindingsPanel from './components/FindingsPanel';
import BudgetBar from './components/BudgetBar';
import EvalDashboard from './components/EvalDashboard';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [prUrl, setPrUrl] = useState('');
  const [runId, setRunId] = useState(null);
  const [isReplay, setIsReplay] = useState(false);
  const [screen, setScreen] = useState('review'); // 'review' | 'eval'
  const [launchError, setLaunchError] = useState('');
  const [initialBudget, setInitialBudget] = useState(null);

  // Seeds the budget bar with today's real cumulative spend on load —
  // without this, a run that's entirely cache hits (zero real LLM
  // calls) would leave the bar stuck at 0 even though tokens were
  // genuinely spent earlier today.
  useEffect(() => {
    fetch(`${API_BASE}/budget`)
      .then((res) => res.json())
      .then(setInitialBudget)
      .catch(() => {});
  }, []);

  const { agentState, agentDetail, workerCounts, events, findings, budget, status, reportText } =
    useEventStream(runId, {
      replay: isReplay,
      apiBase: API_BASE,
      initialBudget,
    });

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
        <span className="app-title">§ REVIEW SWARM</span>
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
      </header>

      {screen === 'review' ? (
        <>
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
              {status === 'streaming' ? 'Running…' : 'Run review'}
            </button>
            <button
              className="pr-replay"
              type="button"
              onClick={() => {
                const id = window.prompt('Enter a recorded run_id to replay:');
                if (id) startReplay(id);
              }}
            >
              ▶ Replay
            </button>
          </form>

          {launchError && <div className="launch-error">{launchError}</div>}

          <main className="columns">
            <section className="col col--graph">
              <AgentGraph agentState={agentState} agentDetail={agentDetail} workerCounts={workerCounts} />
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