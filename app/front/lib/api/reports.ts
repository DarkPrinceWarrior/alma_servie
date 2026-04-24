import { apiGet } from "./client";
import type {
  AnomalyReportAvailability,
  AnomalyType,
  DetectorType,
  PredictedStart,
  ScoreSeries,
} from "./types";

export function reportHtmlUrl(
  anomaly: AnomalyType,
  detector: DetectorType | string,
): string {
  return `/api/reports/${anomaly}/${detector}/html`;
}

export function featureImportanceHtmlUrl(
  anomaly: AnomalyType,
  detector: DetectorType | string,
): string {
  return `/api/reports/${anomaly}/${detector}/feature-importance`;
}

export async function getAvailability(
  anomaly: AnomalyType,
): Promise<AnomalyReportAvailability> {
  return apiGet<AnomalyReportAvailability>(
    `/api/reports/${anomaly}/availability`,
  );
}

export interface ScoresQuery {
  well_id?: string;
  from?: string;
  to?: string;
  limit?: number;
}

export async function getScores(
  anomaly: AnomalyType,
  detector: DetectorType,
  query: ScoresQuery = {},
): Promise<ScoreSeries> {
  const params = new URLSearchParams();
  if (query.well_id) params.set("well_id", query.well_id);
  if (query.from) params.set("from", query.from);
  if (query.to) params.set("to", query.to);
  if (query.limit) params.set("limit", String(query.limit));
  const qs = params.toString();
  return apiGet<ScoreSeries>(
    `/api/reports/${anomaly}/${detector}/scores${qs ? `?${qs}` : ""}`,
  );
}

export async function getPredictedStarts(
  anomaly: AnomalyType,
  detector: DetectorType,
  opts: { well_id?: string; split?: string } = {},
): Promise<PredictedStart[]> {
  const params = new URLSearchParams();
  if (opts.well_id) params.set("well_id", opts.well_id);
  if (opts.split) params.set("split", opts.split);
  const qs = params.toString();
  return apiGet<PredictedStart[]>(
    `/api/reports/${anomaly}/${detector}/starts${qs ? `?${qs}` : ""}`,
  );
}
