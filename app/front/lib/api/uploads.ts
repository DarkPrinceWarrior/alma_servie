import { apiDelete, apiGet, apiPostJson, apiPostMultipart } from "./client";
import type { DetectionRunRead, UploadList, UploadResultBundle } from "./types";

export async function createUpload(
  wellId: string,
  file: File,
  anomalies: readonly string[],
): Promise<DetectionRunRead> {
  const form = new FormData();
  form.append("file", file);
  form.append("well_id", wellId);
  form.append("anomalies", anomalies.join(","));
  return apiPostMultipart<DetectionRunRead>("/api/uploads", form);
}

export async function getUploadResult(
  runId: string,
): Promise<UploadResultBundle> {
  return apiGet<UploadResultBundle>(`/api/uploads/${runId}/result`);
}

export async function listUploads(): Promise<UploadList> {
  return apiGet<UploadList>("/api/uploads");
}

export async function deleteUpload(runId: string): Promise<void> {
  await apiDelete(`/api/uploads/${runId}`);
}

export async function bulkDeleteUploads(
  runIds: string[],
): Promise<{ deleted: number }> {
  return apiPostJson<{ deleted: number }>("/api/uploads/bulk-delete", {
    run_ids: runIds,
  });
}
