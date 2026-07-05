import { useState } from 'react';

// Truncate source text toggle
export function SourceItem({ source, index }) {
  const [expanded, setExpanded] = useState(false);
  const scorePercent = Math.round(source.score * 100);

  return (
    <div className="source-item">
      <div className="source-meta">
        <span className="source-page">
          📄 Page {source.page ?? '?'}
        </span>
        <span className="source-score">
          <span className="score-bar">
            <span className="score-fill" style={{ width: `${scorePercent}%` }} />
          </span>
          {source.score.toFixed(3)}
        </span>
      </div>
      <p className={`source-text${expanded ? ' expanded' : ''}`}>
        {source.text}
      </p>
      <button
        className="source-expand"
        onClick={() => setExpanded(p => !p)}
        aria-expanded={expanded}
      >
        {expanded ? '▲ Show less' : '▼ Show more'}
      </button>
    </div>
  );
}
