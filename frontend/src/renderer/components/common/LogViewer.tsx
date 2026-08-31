import React, { useEffect, useRef } from 'react';

interface LogViewerProps {
  logs: string[];
  autoScroll?: boolean;
  maxHeight?: string;
}

const LogViewer: React.FC<LogViewerProps> = ({ logs, autoScroll = true, maxHeight = '300px' }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  return (
    <div
      ref={containerRef}
      className="overflow-y-auto rounded-lg p-3"
      style={{
        maxHeight,
        backgroundColor: '#0a0a14',
        fontFamily: 'var(--font-mono)',
        fontSize: '12px',
        lineHeight: '1.6',
      }}
    >
      {logs.length === 0 ? (
        <span style={{ color: 'var(--foreground-dim)' }}>暂无日志输出...</span>
      ) : (
        logs.map((line, i) => (
          <div key={i} className="whitespace-pre-wrap" style={{ color: '#a0e0a0' }}>
            {line}
          </div>
        ))
      )}
    </div>
  );
};

export default LogViewer;
