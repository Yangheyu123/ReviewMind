// ============ API Response Wrapper ============
export interface ApiResponse<T> {
  code: number;
  message: string;
  data: T;
}

// ============ Review Job ============
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface ReviewConfig {
  enable_ast: boolean;
  enable_rag: boolean;
  strict_mode: boolean;
}

export interface ReviewProgress {
  step: StepType;
  percent: number;
  message: string;
}

export interface ReviewJob {
  job_id: string;
  pr_url: string;
  status: JobStatus;
  progress: ReviewProgress;
  config: ReviewConfig;
  created_at: string;
  updated_at: string;
  completed_at?: string;
}

export interface CreateJobResponse {
  job_id: string;
  status: JobStatus;
  stream_url: string;
  report_url: string;
}

// ============ PR Info ============
export interface PullRequestInfo {
  owner: string;
  repo: string;
  number: number;
  title: string;
  author: string;
  base_branch: string;
  head_branch: string;
  changed_files: number;
  additions: number;
  deletions: number;
  html_url: string;
}

// ============ Changed File ============
export type FileStatus = 'added' | 'modified' | 'removed' | 'renamed';

export interface ChangedFile {
  filename: string;
  status: FileStatus;
  additions: number;
  deletions: number;
  changes: number;
  patch?: string;
  old_code?: string;
  new_code?: string;
  risk_count?: number;
}

// ============ Changed Symbol ============
export interface ChangedSymbol {
  file: string;
  symbol: string;
  language: string;
  start_line: number;
  end_line: number;
  changed_lines: number[];
  code?: string;
}

// ============ Finding ============
export type RiskLevel = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'WARNING' | 'INFO';
// 后端实际下发 snake_case（如 security_agent）
export type AgentName =
  | 'summary_agent'
  | 'security_agent'
  | 'performance_agent'
  | 'test_agent'
  | 'risk_judge'
  | 'report_agent';

export interface Finding {
  id: string;
  agent: AgentName;
  type: string;
  level: RiskLevel;
  confidence: number;
  file: string;
  line: number;
  symbol?: string;
  description: string;
  suggestion: string;
  code_snippet?: string;
}

// ============ Review Report ============
export type OverallRiskLevel = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'PASS';

export interface RiskStats {
  critical: number;
  high: number;
  medium: number;
  low: number;
  suggestion: number;
}

export interface ReviewReport {
  summary: string;
  risk_level: OverallRiskLevel;
  stats: RiskStats;
  changed_files: ChangedFile[];
  changed_symbols: ChangedSymbol[];
  findings: Finding[];
  review_comment: string;
}

// ============ Complete Job Response ============
export interface JobDetailResponse {
  job_id: string;
  status: JobStatus;
  pr?: PullRequestInfo;
  report?: ReviewReport;
  progress?: ReviewProgress;
  findings?: Finding[];
  created_at: string;
  completed_at?: string;
  updated_at?: string;
  error_message?: string;
}

// ============ Job List ============
export interface JobListItem {
  job_id: string;
  status: JobStatus;
  pr_title: string;
  pr_url: string;
  risk_level: OverallRiskLevel;
  finding_count: number;
  created_at: string;
}

export interface JobListResponse {
  items: JobListItem[];
  page: number;
  page_size: number;
  total: number;
}

// ============ SSE Events ============
export type StepType =
  | 'FETCH_PR'
  | 'DIFF_FILTER'
  | 'DIFF_PARSE'
  | 'AST_CONTEXT'
  | 'SUMMARY_AGENT'
  | 'SECURITY_AGENT'
  | 'PERFORMANCE_AGENT'
  | 'TEST_AGENT'
  | 'RISK_JUDGE'
  | 'REPORT_AGENT'
  // Planner → Executor → Finalizer
  | 'PLANNER'
  | 'PLANNER_DONE'
  | 'EXECUTOR'
  | 'FINALIZER'
  | 'DONE';

export interface ProgressEvent {
  step: StepType;
  percent: number;
  message: string;
}

export interface ChunkEvent {
  target: 'summary' | 'report' | 'files';
  content: string;
}

export interface FindingEvent {
  id: string;
  agent: AgentName;
  file: string;
  line: number;
  symbol?: string;
  level: RiskLevel;
  type: string;
  confidence: number;
  description: string;
  suggestion: string;
}

export interface WarningEvent {
  code: string;
  message: string;
  file?: string;
}

export interface DoneEvent {
  job_id: string;
  status: 'completed' | 'failed' | 'cancelled' | 'error';
  report_url: string;
  total_findings?: number;
  duration_ms?: number;
  error_message?: string;
}

export interface ErrorEvent {
  code: number;
  message: string;
  detail?: string;
}

// ============ Post Comment ============
export interface PostCommentResponse {
  comment_id: number;
  html_url: string;
}

// ============ Merge PR ============
export interface MergeRequest {
  commit_title?: string;
  commit_message?: string;
  merge_method?: 'merge' | 'squash' | 'rebase';
  github_token?: string;
}

export interface MergeResponse {
  merged: boolean;
  message: string;
  sha?: string;
  html_url?: string;
}

// ============ Step Display Info ============
export interface StepInfo {
  key: StepType;
  label: string;
  description: string;
}
