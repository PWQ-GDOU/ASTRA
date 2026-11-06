import React from 'react';
import { useStatisticsStore } from '../stores/statistics.store';
import PageHeader from '../components/common/PageHeader';
import GlassPanel from '../components/common/GlassPanel';
import {
  LineChart,
  AreaChart,
  PieChart,
  RadarChart,
  BarChart,
  ChartSkeleton,
} from '../components/charts';

const Statistics: React.FC = () => {
  const { data, isLoading } = useStatisticsStore();

  return (
    <div className="space-y-5">
      <PageHeader title="统计分析" description="系统训练和推理数据概览" />

      <div className="grid grid-cols-2 gap-5">
        {/* 训练趋势 — 折线图 */}
        <GlassPanel title="训练趋势 (Epoch vs Loss)">
          <div className="p-2">
            {isLoading ? (
              <ChartSkeleton />
            ) : (
              <LineChart
                categories={data?.trainingTrend?.map((p) => `E${p.epoch}`) ?? []}
                series={
                  data?.trainingTrend
                    ? [
                        {
                          name: 'Training Loss',
                          data: data.trainingTrend.map((p) => p.loss),
                        },
                        ...(data.trainingTrend.some((p) => p.valLoss != null)
                          ? [
                              {
                                name: 'Validation Loss',
                                data: data.trainingTrend.map((p) => p.valLoss ?? 0),
                              } as const,
                            ]
                          : []),
                      ]
                    : []
                }
                smooth
                emptyText="加载统计数据显示训练趋势"
              />
            )}
          </div>
        </GlassPanel>

        {/* 推理准确率 — 面积图 */}
        <GlassPanel title="推理准确率趋势">
          <div className="p-2">
            {isLoading ? (
              <ChartSkeleton />
            ) : (
              <AreaChart
                categories={data?.inferenceAccuracyTrend?.map((p) => p.date) ?? []}
                data={data?.inferenceAccuracyTrend?.map((p) => p.accuracy) ?? []}
                name="准确率"
                emptyText="加载统计数据显示准确率趋势"
              />
            )}
          </div>
        </GlassPanel>

        {/* 数据集分布 — 饼图 */}
        <GlassPanel title="数据集分布">
          <div className="p-2">
            {isLoading ? (
              <ChartSkeleton />
            ) : (
              <PieChart
                data={
                  data?.datasetDistribution?.map((d) => ({
                    name: d.name,
                    value: d.count,
                  })) ?? []
                }
                emptyText="加载统计数据显示数据集分布"
              />
            )}
          </div>
        </GlassPanel>

        {/* 模型性能对比 — 雷达图 */}
        <GlassPanel title="模型性能对比">
          <div className="p-2">
            {isLoading ? (
              <ChartSkeleton />
            ) : (
              <RadarChart
                indicators={
                  data?.modelPerformance?.length
                    ? [
                        { name: '准确率', max: 100 },
                        { name: '推理速度', max: 500 },
                        { name: '模型大小', max: 1000 },
                        { name: '参数量', max: 2000 },
                      ]
                    : []
                }
                series={
                  data?.modelPerformance?.length
                    ? data.modelPerformance.map((m) => ({
                        name: m.name,
                        data: [m.accuracy, m.speed, m.size, m.params],
                      }))
                    : []
                }
                emptyText="加载统计数据显示模型性能对比"
              />
            )}
          </div>
        </GlassPanel>

        {/* 推理任务统计 — 柱状图 */}
        <GlassPanel title="推理任务统计">
          <div className="p-2">
            {isLoading ? (
              <ChartSkeleton />
            ) : (
              <BarChart
                categories={['已完成', '失败', '运行中', '等待中']}
                data={
                  data?.taskStats
                    ? [
                        data.taskStats.completed,
                        data.taskStats.failed,
                        data.taskStats.running,
                        data.taskStats.pending,
                      ]
                    : []
                }
                name="任务数"
                emptyText="加载统计数据显示任务统计"
              />
            )}
          </div>
        </GlassPanel>

        {/* 模型状态分布 — 饼图 (环形) */}
        <GlassPanel title="模型状态分布">
          <div className="p-2">
            {isLoading ? (
              <ChartSkeleton />
            ) : (
              <PieChart
                data={
                  data?.modelStatusDist?.map((d) => ({
                    name: d.status,
                    value: d.count,
                  })) ?? []
                }
                doughnut
                emptyText="加载统计数据显示模型状态分布"
              />
            )}
          </div>
        </GlassPanel>
      </div>
    </div>
  );
};

export default Statistics;
