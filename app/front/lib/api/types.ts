// TS types mirroring pydantic schemas in app/back/src/back/api/*/schemas.py.
// Keep in sync manually; a codegen from openapi.json is a follow-up.

export type AnomalyType = "negermet" | "pritok" | "salt";
export type SplitType = "train" | "test";
export type DetectorType = "paano_shared";
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

export interface TimePoint {
  t: string;
  v: number;
}

export interface TelemetryChannel {
  name: string;
  points: TimePoint[];
}

export interface AnomalyInterval {
  start: string;
  end: string;
  interval_idx: number;
  split: string;
}

export interface PredictedOnset {
  t: string;
  split: string;
}

export interface IntervalResult {
  interval_idx: number;
  actual_start: string;
  actual_end: string;
  detected_time: string | null;
  delay_hours: number | null;
  status: string;
  split: string;
  data_start: string | null;
  data_end: string | null;
}

export interface WellSeriesResponse {
  well_id: string;
  anomaly: string;
  detector: string;
  n_points_raw: number;
  n_points_downsampled: number;
  time_start: string | null;
  time_end: string | null;
  score: TimePoint[];
  paano_short: TimePoint[];
  paano_long: TimePoint[];
  telemetry: TelemetryChannel[];
  intervals: AnomalyInterval[];
  predicted_starts: PredictedOnset[];
  results: IntervalResult[];
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface FeatureImportanceResponse {
  well_id: string;
  anomaly: string;
  detector: string;
  items: FeatureImportanceItem[];
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

// uploads (unlabeled well Excel -> inference across all 3 anomaly classes)
export interface UploadScorePoint {
  t: string;
  score: number;
}

export interface UploadAnomalyResult {
  anomaly: AnomalyType;
  status: "succeeded" | "failed" | "pending";
  well_id: string | null;
  detector: string | null;
  n_points: number | null;
  n_detected: number | null;
  detected_starts: string[];
  score_min: number | null;
  score_median: number | null;
  score_max: number | null;
  time_start: string | null;
  time_end: string | null;
  score_series: UploadScorePoint[];
  error: string | null;
}

export interface UploadResultBundle {
  run_id: string;
  well_id: string;
  status: RunStatus;
  n_done: number;
  n_total: number;
  results: UploadAnomalyResult[];
}
