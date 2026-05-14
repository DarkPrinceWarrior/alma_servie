import { apiGet, apiPostMultipart } from "./client";
import type { AnomalyType, DetectionRunRead, UploadResult } from "./types";

export async function createUpload(
  anomaly: AnomalyType,
  wellId: string,
  file: File,
): Promise<DetectionRunRead> {
  const form = new FormData();
  form.append("file", file);
  form.append("anomaly", anomaly);
  form.append("well_id", wellId);
  return apiPostMultipart<DetectionRunRead>("/api/uploads", form);
}

export async function getUpload(runId: string): Promise<DetectionRunRead> {
  return apiGet<DetectionRunRead>(`/api/uploads/${runId}`);
}

export async function getUploadResult(runId: string): Promise<UploadResult> {
  return apiGet<UploadResult>(`/api/uploads/${runId}/result`);
}
