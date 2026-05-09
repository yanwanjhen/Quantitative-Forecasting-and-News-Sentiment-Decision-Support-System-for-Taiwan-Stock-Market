import type { ChatMessage, InvestorProfile, SessionsResponse, StreamEvent } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export async function loadSessions(userId?: string): Promise<SessionsResponse> {
  const suffix = userId ? `?user_id=${encodeURIComponent(userId)}` : "";
  const response = await fetch(`${API_BASE}/api/sessions${suffix}`);
  if (!response.ok) throw new Error("Unable to load sessions");
  return response.json();
}

export async function createSession(userId: string): Promise<{ session_id: string; messages: ChatMessage[] }> {
  const response = await fetch(`${API_BASE}/api/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId }),
  });
  if (!response.ok) throw new Error("Unable to create session");
  return response.json();
}

export async function deleteSession(
  sessionId: string,
  userId: string,
): Promise<{ current_session: string; sessions: SessionsResponse["sessions"] }> {
  const suffix = `?user_id=${encodeURIComponent(userId)}`;
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}${suffix}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error("Unable to delete session");
  return response.json();
}

export async function updateProfile(userId: string, profile: InvestorProfile): Promise<InvestorProfile> {
  const response = await fetch(`${API_BASE}/api/profile`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, profile }),
  });
  if (!response.ok) throw new Error("Unable to update profile");
  const payload = await response.json();
  return payload.investor_profile;
}

export async function streamMessage(
  sessionId: string,
  payload: { user_id: string; content: string; profile: InvestorProfile; model?: string },
  apiKey: string,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/messages/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { "X-Groq-API-Key": apiKey } : {}),
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new Error(text || "Unable to stream message");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const lines = part.split("\n");
      const eventLine = lines.find((line) => line.startsWith("event: "));
      const dataLine = lines.find((line) => line.startsWith("data: "));
      if (!eventLine || !dataLine) continue;
      onEvent({
        event: eventLine.slice(7),
        data: JSON.parse(dataLine.slice(6)),
      } as StreamEvent);
    }
  }
}
