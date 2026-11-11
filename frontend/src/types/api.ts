export interface ChatRequest {
  message: string;
  user_id: string;
  conv_id?: string | null;
}

export interface ChatResponse {
  conv_id: string;
  response: string;
  intent: string;
  agent_type: string;
  escalated: boolean;
  latency_ms: number;
  knowledge_used: boolean;
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  streaming?: boolean;
  meta?: Pick<
    ChatResponse,
    "intent" | "agent_type" | "escalated" | "latency_ms" | "knowledge_used"
  >;
}

export type StreamMode = "sse" | "fallback";

export interface KnowledgeDocument {
  title: string;
  content: string;
}

export interface KnowledgeImportResponse {
  message: string;
  added_chunks: number;
  total_chunks: number;
  document_id?: string;
  pages?: number;
  native_pages?: number;
  ocr_pages?: number;
  hybrid_pages?: number;
  quality_score?: number;
  warnings?: string[];
}

export interface KnowledgeStats {
  total_chunks: number;
}

export interface SearchResultItem {
  title: string;
  content: string;
  score: number;
  chunk?: number;
  fallback?: boolean;
  error?: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
  reranked: boolean;
}

export interface AgentStats {
  total: number;
  success_rate: number;
  avg_ms: number;
  monitor_penalty: number;
  routing_score: number;
}

export interface HealthResponse {
  status: string;
  agents: Record<string, AgentStats>;
}

export interface MonitorAlert {
  id?: string;
  severity?: string;
  message?: string;
  metric?: string;
  value?: number;
  threshold?: number;
  timestamp?: string;
  resolved?: boolean;
  [key: string]: unknown;
}

export interface MonitorSuggestion {
  title: string;
  action: string;
  priority: number;
}

export interface MonitorResponse {
  agent_stats: Record<string, AgentStats>;
  tool_stats: Record<string, Record<string, unknown>>;
  active_alerts: MonitorAlert[];
  suggestions: MonitorSuggestion[];
}

export interface SkillSummary {
  [key: string]: unknown;
}

export interface EvalIntentInput {
  message: string;
  expected_intent: string;
  context?: Record<string, unknown>;
}

export interface EvalDialogInput {
  question?: string;
  turns?: string[];
  user_id?: string;
  conv_id?: string;
}

export interface EvalRunInput {
  intent_cases?: EvalIntentInput[];
  dialog_cases?: EvalDialogInput[];
}

export interface EvalResult {
  test_id: string;
  passed: boolean;
  scores: Record<string, number>;
  detail: string;
  metadata: Record<string, unknown>;
}

export interface EvalReport {
  pass_rate: number;
  total: number;
  passed: number;
  avg_scores: Record<string, number>;
  regressions: string[];
  recommendations: string[];
  results: EvalResult[];
}
