import { apiGet } from "./client";
import type {
  AnomalyType,
  WellDetail,
  WellInterval,
  WellSummary,
} from "./types";

export async function listWells(anomaly: AnomalyType): Promise<WellSummary[]> {
  return apiGet<WellSummary[]>(`/api/wells?anomaly=${anomaly}`);
}

export async function getWell(
  anomaly: AnomalyType,
  wellId: string,
): Promise<WellDetail> {
  return apiGet<WellDetail>(
    `/api/wells/${encodeURIComponent(wellId)}?anomaly=${anomaly}`,
  );
}

export async function getWellIntervals(
  anomaly: AnomalyType,
  wellId: string,
): Promise<WellInterval[]> {
  return apiGet<WellInterval[]>(
    `/api/wells/${encodeURIComponent(wellId)}/intervals?anomaly=${anomaly}`,
  );
}
