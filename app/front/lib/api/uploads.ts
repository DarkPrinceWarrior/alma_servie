import { apiGet, apiPostMultipart } from "./client";
import type { DetectionRunRead, UploadResultBundle } from "./types";

export async function createUpload(
  wellId: string,
  file: File,
): Promise<DetectionRunRead> {
  const form = new FormData();
  form.append("file", file);
  form.append("well_id", wellId);
  return apiPostMultipart<DetectionRunRead>("/api/uploads", form);
}

export async function getUploadResult(
  runId: string,
): Promise<UploadResultBundle> {
  return apiGet<UploadResultBundle>(`/api/uploads/${runId}/result`);
}
