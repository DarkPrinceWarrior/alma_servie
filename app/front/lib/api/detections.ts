import { apiGet, apiPostJson } from "./client";
import type {
  AnomalyType,
  DetectionRunList,
  DetectionRunRead,
  DetectorType,
  RunStatus,
} from "./types";

export async function launchDetection(
  anomaly: AnomalyType,
  detector: DetectorType,
): Promise<DetectionRunRead> {
  return apiPostJson<DetectionRunRead>("/api/detections", {
    anomaly,
    detector,
  });
}

export interface ListRunsQuery {
  anomaly?: AnomalyType;
  detector?: DetectorType;
  status?: RunStatus;
  limit?: number;
  offset?: number;
}

export async function listRuns(
  query: ListRunsQuery = {},
): Promise<DetectionRunList> {
  const params = new URLSearchParams();
  if (query.anomaly) params.set("anomaly", query.anomaly);
  if (query.detector) params.set("detector", query.detector);
  if (query.status) params.set("status", query.status);
  if (query.limit !== undefined) params.set("limit", String(query.limit));
  if (query.offset !== undefined) params.set("offset", String(query.offset));
  const qs = params.toString();
  return apiGet<DetectionRunList>(`/api/detections${qs ? `?${qs}` : ""}`);
}

export async function getRun(runId: string): Promise<DetectionRunRead> {
  return apiGet<DetectionRunRead>(`/api/detections/${runId}`);
}
