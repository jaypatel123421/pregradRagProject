import { useState, useEffect, useRef, useCallback } from 'react';
import { getStatus, ingestPDF, queryRAG } from './api';
import { SourceItem } from './SourceItem';

const PDF_PATH = '/Users/jaypatel/Downloads/DaVinci-Resolve-16_Beginners-Guide.pdf';

const SUGGESTED_QUESTIONS = [
  'How do I import media into DaVinci Resolve?',
  'What is the Color page used for?',
  'How do I add transitions between clips?',
  'What is a timeline in DaVinci Resolve?',
  'How do I export a finished video?',
  'What are nodes in the color grading workflow?',
];

// Status indicator states
const STATUS_STATES = {
  loading: { label: 'Connecting…', cls: 'loading' },
  ready: { label: 'Index Ready', cls: 'ready' },
  empty: { label: 'Not Indexed', cls: '' },
  error: { label: 'Error', cls: 'error' },
};

export default function App() {
  const [status, setStatus] = useState('loading');
  const [chunksIndexed, setChunksIndexed] = useState(0);
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState(null);
  const [question, setQuestion] = useState('');
  const [queryLoading, setQueryLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [queryError, setQueryError] = useState(null);
  const textareaRef = useRef(null);
  const resultRef = useRef(null);

  // ------------------------------------------------------------------
  // Status polling on mount
  // ------------------------------------------------------------------
  const fetchStatus = useCallback(async () => {
    try {
      const data = await getStatus();
      setStatus(data.ready ? 'ready' : 'empty');
      setChunksIndexed(data.chunks_indexed ?? 0);
    } catch {
      setStatus('error');
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // ------------------------------------------------------------------
  // Ingest
  // ------------------------------------------------------------------
  const handleIngest = async (force = false) => {
    setIngesting(true);
    setIngestError(null);
    try {
      const data = await ingestPDF(PDF_PATH, force);
      setChunksIndexed(data.chunks_indexed);
      setStatus('ready');
    } catch (err) {
      setIngestError(err.message);
      setStatus('error');
    } finally {
      setIngesting(false);
    }
  };

  // ------------------------------------------------------------------
  // Query
  // ------------------------------------------------------------------
  const handleQuery = async (q) => {
    const questionText = (q || question).trim();
    if (!questionText) return;

    setQuestion(questionText);
    setQueryLoading(true);
    setQueryError(null);
    setResult(null);

    try {
      const data = await queryRAG(questionText, 5);
      setResult(data);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (err) {
      setQueryError(err.message);
    } finally {
      setQueryLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  const handleChipClick = (q) => {
    setQuestion(q);
    handleQuery(q);
  };

  // ------------------------------------------------------------------
  // Auto-resize textarea
  // ------------------------------------------------------------------
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [question]);

  const statusState = STATUS_STATES[status] || STATUS_STATES.loading;
  const isReady = status === 'ready';

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="header-brand">
            <div className="brand-icon">🎬</div>
            <div>
              <div className="brand-name">DaVinci Resolve AI Guide</div>
              <div className="brand-sub">Powered by Qdrant + OpenAI</div>
            </div>
          </div>
          <div className="header-status" id="status-indicator">
            <span className={`status-dot ${statusState.cls}`} />
            {statusState.label}
            {isReady && chunksIndexed > 0 && (
              <span style={{ color: 'var(--text-muted)' }}>· {chunksIndexed} chunks</span>
            )}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="main">
        {/* Hero */}
        <section className="hero">
          <div className="hero-badge">🎯 RAG · Qdrant · Gemini</div>
          <h1>
            Ask anything about <span>DaVinci Resolve</span>
          </h1>
          <p>
            Instant answers sourced directly from the official Beginner's Guide PDF —
            with page references and similarity scores.
          </p>
        </section>

        {/* Setup Card — shown when not indexed */}
        {status === 'empty' && !ingesting && (
          <div className="setup-card" id="setup-card">
            <div className="setup-card-header">
              <div className="setup-icon">📥</div>
              <div>
                <h2>Index the Guide</h2>
              </div>
            </div>
            <p>
              The PDF needs to be embedded and stored in Qdrant before you can ask questions.
              Click below to start ingestion — this only happens once.
            </p>
            <div className="setup-path">{PDF_PATH}</div>
            {ingestError && (
              <div className="error-banner">
                <span className="error-icon">⚠️</span>
                <span>{ingestError}</span>
              </div>
            )}
            <button
              id="ingest-btn"
              className="ingest-btn"
              onClick={() => handleIngest(false)}
              disabled={ingesting}
            >
              🚀 Start Ingestion
            </button>
          </div>
        )}

        {/* Ingestion in progress */}
        {ingesting && (
          <div className="ingest-progress">
            <div className="spinner" />
            Embedding and uploading PDF chunks to Qdrant… this may take a few minutes.
          </div>
        )}

        {/* Ingest error after it was previously ready */}
        {ingestError && status !== 'empty' && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            <span>{ingestError}</span>
          </div>
        )}

        {/* Search */}
        {isReady && (
          <>
            <section className="search-section" id="search-section">
              <form
                className="search-form"
                onSubmit={(e) => { e.preventDefault(); handleQuery(); }}
              >
                <div className="search-input-wrapper">
                  <span className="search-input-icon">🔍</span>
                  <textarea
                    ref={textareaRef}
                    id="question-input"
                    className="search-input"
                    placeholder="Ask a question about DaVinci Resolve…"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={queryLoading}
                    rows={1}
                    aria-label="Question input"
                  />
                </div>
                <button
                  id="ask-btn"
                  type="submit"
                  className="search-btn"
                  disabled={queryLoading || !question.trim()}
                >
                  {queryLoading ? (
                    <><div className="spinner dark sm" /> Thinking…</>
                  ) : (
                    <>Ask ✨</>
                  )}
                </button>
              </form>
              <div className="search-hint">
                <span>Press <kbd>Enter</kbd> to ask &nbsp;·&nbsp; <kbd>Shift</kbd>+<kbd>Enter</kbd> for new line</span>
              </div>
            </section>

            {/* Suggestions */}
            {!result && !queryLoading && (
              <div className="suggestions">
                <div className="suggestions-label">Try asking</div>
                <div className="suggestion-chips">
                  {SUGGESTED_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      className="chip"
                      onClick={() => handleChipClick(q)}
                      type="button"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* Loading skeleton */}
        {queryLoading && (
          <div className="loading-card" aria-live="polite" aria-label="Loading answer">
            <div className="skeleton skeleton-header" />
            <div className="skeleton skeleton-line" />
            <div className="skeleton skeleton-line short" />
            <div className="skeleton skeleton-line shorter" />
            <div style={{ marginTop: 8 }}>
              <div className="skeleton skeleton-header" style={{ width: '30%', marginBottom: 10 }} />
              <div className="skeleton skeleton-line" />
              <div className="skeleton skeleton-line short" />
            </div>
          </div>
        )}

        {/* Query error */}
        {queryError && !queryLoading && (
          <div className="error-banner" role="alert">
            <span className="error-icon">⚠️</span>
            <span>{queryError}</span>
          </div>
        )}

        {/* Answer card */}
        {result && !queryLoading && (
          <div className="answer-card" ref={resultRef} id="answer-card">
            <div className="answer-header">
              <div>
                <div className="answer-question-label">Your question</div>
                <div className="answer-question">{result.question}</div>
              </div>
            </div>
            <div className="answer-body">
              <div>
                <div className="answer-section-label">Answer</div>
                <p className="answer-text">{result.answer}</p>
              </div>

              {result.sources?.length > 0 && (
                <div>
                  <div className="answer-section-label">
                    Source Excerpts ({result.sources.length})
                  </div>
                  <div className="sources-list">
                    {result.sources.map((src, i) => (
                      <SourceItem key={i} source={src} index={i} />
                    ))}
                  </div>
                </div>
              )}

              {isReady && (
                <button
                  className="chip"
                  style={{ alignSelf: 'flex-start' }}
                  onClick={() => handleIngest(true)}
                  disabled={ingesting}
                  type="button"
                  id="re-ingest-btn"
                >
                  🔄 Re-index PDF
                </button>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <span className="footer-tag">🎬 DaVinci Resolve Beginner's Guide</span>
        <span className="footer-tag">⚡ Qdrant Vector DB</span>
        <span className="footer-tag">🤖 OpenAI</span>
      </footer>
    </div>
  );
}
