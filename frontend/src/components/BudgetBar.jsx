/**
 * BudgetBar — the signature element per the plan: a live token budget
 * bar along the bottom that fills as the run progresses and visibly
 * marks when the pipeline is throttled waiting on rate limits.
 */

export default function BudgetBar({ budget }) {
  const { tpdSpent, tpdLimit } = budget;
  const pct = tpdLimit > 0 ? Math.min((tpdSpent / tpdLimit) * 100, 100) : 0;

  const barColor = pct > 90 ? 'var(--blocker)' : pct > 70 ? 'var(--major)' : 'var(--pass)';

  return (
    <div className="budget-bar">
      <span className="budget-bar__label">TPD</span>
      <div className="budget-bar__track">
        <div
          className="budget-bar__fill"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
      <span className="budget-bar__value">
        {tpdSpent.toLocaleString()} / {tpdLimit.toLocaleString()}
      </span>
    </div>
  );
}