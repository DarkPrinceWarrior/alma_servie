// TS types mirroring pydantic schemas in app/back/src/back/api/*/schemas.py.
// Keep in sync manually; a codegen from openapi.json is a follow-up.

export type AnomalyType = "negermet" | "pritok" | "salt";
export type SplitType = "train" | "test";
export type DetectorType =
  | "pca_spe"
  | "paano_feat"
  | "paano_shared"
  | "ensemble";
export type RunStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

// wells
export interface WellSummary {
  well_id: string;
  anomaly: AnomalyType;
  split: SplitType;
  n_intervals: number;
  data_start: string;
  data_end: string;
}

export interface WellInterval {
  interval_idx: number;
  start_date: string;
  end_date: string;
  data_start: string;
  data_end: string;
  split: SplitType;
}

export interface WellDetail extends WellSummary {
  intervals: WellInterval[];
}

// reports
export interface DetectorAvailability {
  detector: string;
  has_report: boolean;
  has_feature_importance: boolean;
}

export interface AnomalyReportAvailability {
  anomaly: string;
  detectors: DetectorAvailability[];
  best_detector: string | null;
  has_any_report: boolean;
}

export interface ScorePoint {
  t: string;
  score: number;
  split: string;
}

export interface ScoreSeries {
  well_id: string;
  anomaly: string;
  detector: string;
  n_points: number;
  n_downsampled: number;
  points: ScorePoint[];
}

export interface PredictedStart {
  well_id: string;
  detected_time: string;
  split: string;
}

// detections
export interface DetectionRunRead {
  id: string;
  anomaly: string;
  detector: string;
  status: RunStatus;
  command: string;
  exit_code: number | null;
  stdout_tail: string | null;
  summary_json: Record<string, unknown> | null;
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface DetectionRunList {
  items: DetectionRunRead[];
  total: number;
}

// auth / users
export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface UserRead {
  id: string;
  email: string;
  is_active: boolean;
  roles: string[];
  created_at: string;
  updated_at: string;
}
