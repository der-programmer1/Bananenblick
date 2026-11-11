import type { Thresholds } from "./detection";
export type PerClassMetrics = {
  class: string;
  support: number;
  precision: number;
  recall: number;
  visibleRecall: number;
  visibleTruePositive: number;
  uncertain: number;
};
export type ModelCard = {
  inputSize: number;
  thresholds: Thresholds;
  individualBenchmark?: {
    precision: number;
    recall: number;
    passed: boolean;
    positiveScenes: number;
    negativeScenes: number;
  };
  metrics: {
    perClass: PerClassMetrics[];
    testImages: number;
    visibleRecall: number;
    uncertainCount: number;
  };
};
export type EvaluationReport = {
  test: { perClass: PerClassMetrics[]; visibleRecall: number; uncertainCount: number };
};
