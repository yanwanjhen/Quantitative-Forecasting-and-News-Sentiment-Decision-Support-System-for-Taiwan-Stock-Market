export type Role = "user" | "assistant";

export interface InvestorProfile {
  style: "保守" | "穩健" | "積極";
  risk_tolerance: "low" | "medium" | "high";
  max_loss_pct: number;
}

export interface DashboardData {
  ticker: string;
  company_name: string;
  stock_data: Record<string, unknown>;
  news_data: Record<string, unknown>;
  quant_data: Record<string, unknown>;
  chart_data?: Array<Record<string, string | number>>;
  risk: string;
  investor_profile: InvestorProfile;
}

export interface AnalysisContext extends Partial<DashboardData> {
  mode?: "stock" | "compare";
  heading?: string;
  compared_names?: string;
  stocks?: Array<{ ticker?: string; company_name?: string }>;
  compare_rows?: Array<Record<string, unknown>>;
  recommendation?: string;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  created_at: number;
  dashboard_data?: DashboardData | null;
  analysis_context?: AnalysisContext | null;
}

export interface ChatSession {
  session_id: string;
  title: string;
  messages: ChatMessage[];
}

export interface SessionsResponse {
  user_id: string;
  current_session: string;
  investor_profile: InvestorProfile;
  sessions: ChatSession[];
}

export type StreamEvent =
  | { event: "message"; data: { message: ChatMessage } }
  | { event: "status"; data: { text: string } }
  | { event: "token"; data: { text: string } }
  | { event: "dashboard"; data: { dashboard_data: DashboardData } }
  | { event: "done"; data: { message: ChatMessage } }
  | { event: "error"; data: { message: string } };
