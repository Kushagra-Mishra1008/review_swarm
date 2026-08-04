/**
 * EventStream — scrolling, monospace-aligned log of every event as it
 * arrives: node starts/completions, cache hits, throttle waits, worker
 * spawns. Auto-scrolls to the latest event unless the user has
 * scrolled up to read history.
 */

import { useEffect, useRef, useState } from 'react';

function formatTime(unixSeconds) {
  const d = new Date(unixSeconds * 1000);
  return d.toTimeString().slice(0, 8);
}

function describeEvent(event) {
  const { event_type, data } = event;
  switch (event_type) {
    case 'node_start':
      return `→ ${data.node}`;
    case 'node_complete':
      return `✓ ${data.node}${data.finding_count !== undefined ? ` (${data.finding_count} findings)` : ''}`;
    case 'cache_hit':
      return `⊙ cache hit — ${data.model}`;
    case 'throttle_wait':
      return `⏸ tpm wait ${data.seconds}s — ${data.model}`;
    case 'llm_call_start':
      return `↗ llm call — ${data.model}`;
    case 'llm_call_complete':
      return `↙ llm call complete — ${data.tokens} tok (${data.tpd_spent} tpd total)`;
    case 'worker_spawn':
      return `⚬ worker spawned — ${data.file} (${data.specialist})`;
    case 'worker_complete':
      return `⚭ worker done — ${data.file} (${data.finding_count} findings)`;
    case 'finding':
      return `▲ ${data.severity?.toUpperCase()} — ${data.file}:${data.line}`;
    case 'run_complete':
      return `■ run complete`;
    default:
      return event_type;
  }
}

export default function EventStream({ events }) {
  const bodyRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [events, autoScroll]);

  function handleScroll() {
    const el = bodyRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 20;
    setAutoScroll(atBottom);
  }

  return (
    <div className="panel event-stream">
      <div className="panel__header">EVENT STREAM</div>
      <div className="event-stream__body" ref={bodyRef} onScroll={handleScroll}>
        {events.length === 0 && <div className="event-stream__empty">Waiting for events…</div>}
        {events.map((event, i) => (
          <div key={i} className={`event-line event-line--${event.event_type}`}>
            <span className="event-line__time">{formatTime(event.timestamp)}</span>
            <span className="event-line__text">{describeEvent(event)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}