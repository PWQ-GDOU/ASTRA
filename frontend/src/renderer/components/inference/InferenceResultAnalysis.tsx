import React, { useMemo } from 'react';
import type { InferenceResult } from '../../stores/inference.store';
import GlassPanel from '../common/GlassPanel';
import DataTable, { type Column } from '../common/DataTable';
import LineChart, { type LineChartSeries } from '../charts/LineChart';

interface ResultRow {
  id: string;
  unit: number;
  prediction: number;
  target?: number;
  error?: number;
  absoluteError?: number;
}

const DETAIL_LIMIT = 20;

const formatNumber = (value: number | undefined, digits = 3) =>
  value === undefined ? '—' : value.toFixed(digits);

const InferenceResultAnalysis: React.FC<{ result: InferenceResult }> = ({ result }) => {
  const rows = useMemo<ResultRow[]>(
    () =>
      result.units.map((unit, index) => {
        const prediction = result.predictions[index];
        const target = result.targets?.[index];
        const error = target === undefined ? undefined : prediction - target;
        return {
          id: `${unit}-${index}`,
          unit,
          prediction,
          target,
          error,
          absoluteError: error === undefined ? undefined : Math.abs(error),
        };
      }),
    [result.predictions, result.targets, result.units],
  );

  const errorRows = rows.filter(
    (row): row is ResultRow & { absoluteError: number } => row.absoluteError !== undefined,
  );
  const maxAbsoluteError = errorRows.length
    ? errorRows.reduce((maximum, row) => Math.max(maximum, row.absoluteError), 0)
    : undefined;
  const meanAbsoluteError = errorRows.length
    ? errorRows.reduce((total, row) => total + row.absoluteError, 0) / errorRows.length
    : undefined;

  const chartSeries = useMemo<LineChartSeries[]>(() => {
    const series: LineChartSeries[] = [
      { name: '预测 RUL', data: result.predictions, color: '#6d7cff' },
    ];
    if (result.targets) {
      series.push({ name: '真实 RUL', data: result.targets, color: '#55d6be' });
    }
    return series;
  }, [result.predictions, result.targets]);

  const columns = useMemo<Column<ResultRow>[]>(
    () => [
      { key: 'unit', label: '设备单元', render: (row) => String(row.unit) },
      {
        key: 'prediction',
        label: '预测 RUL',
        align: 'right',
        render: (row) => <span className="font-mono">{formatNumber(row.prediction)}</span>,
      },
      {
        key: 'target',
        label: '真实 RUL',
        align: 'right',
        render: (row) => <span className="font-mono">{formatNumber(row.target)}</span>,
      },
      {
        key: 'error',
        label: '误差（预测 - 真实）',
        align: 'right',
        render: (row) => <span className="font-mono">{formatNumber(row.error)}</span>,
      },
      {
        key: 'absoluteError',
        label: '绝对误差',
        align: 'right',
        render: (row) => <span className="font-mono">{formatNumber(row.absoluteError)}</span>,
      },
    ],
    [],
  );

  const summary = [
    ['预测 RUL', formatNumber(result.predictions[0], 2)],
    ['RMSE', formatNumber(result.rmse)],
    ['MAE', formatNumber(result.mae)],
    ['NASA Score', formatNumber(result.score, 2)],
    ['最大绝对误差', formatNumber(maxAbsoluteError)],
    ['样本平均绝对误差', formatNumber(meanAbsoluteError)],
    ['样本数', String(result.units.length)],
    ['耗时', `${result.seconds.toFixed(2)} s`],
  ];

  return (
    <section aria-label="推理结果分析" className="space-y-5">
      <GlassPanel title="推理结果">
        <div aria-live="polite" className="p-4 grid grid-cols-2 md:grid-cols-4 gap-4">
          {summary.map(([label, value]) => (
            <div
              key={label}
              className="p-4 rounded-lg text-center"
              style={{ background: 'var(--surface)', border: '1px solid var(--border-light)' }}
            >
              <div className="text-xs mb-1" style={{ color: 'var(--foreground-muted)' }}>
                {label}
              </div>
              <div className="text-2xl font-bold font-mono" style={{ color: 'var(--foreground)' }}>
                {value}
              </div>
            </div>
          ))}
        </div>
      </GlassPanel>

      <GlassPanel title="预测与真实 RUL">
        <div
          className="p-4"
          role="img"
          aria-label={
            result.targets ? '各设备单元预测 RUL 与真实 RUL 趋势图' : '各设备单元预测 RUL 趋势图'
          }
        >
          <LineChart
            categories={result.units}
            series={chartSeries}
            height={300}
            smooth={result.units.length > 20}
          />
          {!result.targets && (
            <p className="mt-3 text-xs" style={{ color: 'var(--foreground-muted)' }}>
              当前结果没有真实标签，无法计算逐样本误差。
            </p>
          )}
        </div>
      </GlassPanel>

      <GlassPanel title="样本误差明细">
        <div className="overflow-x-auto">
          <DataTable columns={columns} data={rows.slice(0, DETAIL_LIMIT)} />
        </div>
        {rows.length > DETAIL_LIMIT && (
          <p className="px-4 pb-3 text-xs" style={{ color: 'var(--foreground-muted)' }}>
            显示前 {DETAIL_LIMIT} 个样本，共 {rows.length} 个。
          </p>
        )}
      </GlassPanel>
    </section>
  );
};

export default InferenceResultAnalysis;
