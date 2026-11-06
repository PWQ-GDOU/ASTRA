import React from 'react';
import type { ModelPerformance } from '../../stores/model.store';

const formatMetric = (value: number | undefined, digits = 3) =>
  value === undefined ? '—' : value.toFixed(digits);

const ModelPerformancePanel: React.FC<{ performance: ModelPerformance }> = ({ performance }) => {
  const evaluatedAt = new Date(performance.evaluatedAt);
  const evaluatedLabel = Number.isNaN(evaluatedAt.getTime())
    ? performance.evaluatedAt
    : evaluatedAt.toLocaleString('zh-CN', { hour12: false });

  const metrics = [
    ['RMSE', formatMetric(performance.rmse)],
    ['MAE', formatMetric(performance.mae)],
    ['NASA Score', formatMetric(performance.score, 2)],
    ['推理耗时', `${performance.seconds.toFixed(2)} s`],
  ];

  return (
    <section
      aria-labelledby="latest-model-performance"
      className="p-4 rounded-lg space-y-3"
      style={{ background: 'var(--surface)', border: '1px solid var(--border-light)' }}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3
            id="latest-model-performance"
            className="text-sm font-semibold"
            style={{ color: 'var(--foreground)' }}
          >
            最近实测效果
          </h3>
          <p className="mt-1 text-xs" style={{ color: 'var(--foreground-muted)' }}>
            <span>{performance.datasetName}</span>
            {' · '}
            <span>{performance.sampleCount} 个样本</span>
          </p>
        </div>
        <time
          dateTime={performance.evaluatedAt}
          className="text-xs"
          style={{ color: 'var(--foreground-dim)' }}
        >
          {evaluatedLabel}
        </time>
      </div>
      <dl className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {metrics.map(([label, value]) => (
          <div key={label}>
            <dt className="text-xs" style={{ color: 'var(--foreground-muted)' }}>
              {label}
            </dt>
            <dd
              className="mt-1 text-base font-semibold font-mono"
              style={{ color: 'var(--foreground)' }}
            >
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
};

export default ModelPerformancePanel;
