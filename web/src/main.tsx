import React, { FormEvent, useEffect, useRef, useState, useTransition } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bot,
  CircleAlert,
  Eye,
  EyeOff,
  KeyRound,
  Copy,
  Menu,
  MessageSquarePlus,
  PanelLeftClose,
  X,
  Square,
  Send,
  Settings2,
  SlidersHorizontal,
} from "lucide-react";
import { LineChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { createSession, getApiBaseUrl, loadSessions, streamMessage, updateProfile } from "./api";
import type { AnalysisContext, ChatMessage, ChatSession, DashboardData, InvestorProfile, StreamEvent } from "./types";
import "./styles.css";

const riskByStyle: Record<InvestorProfile["style"], InvestorProfile["risk_tolerance"]> = {
  保守: "low",
  穩健: "medium",
  積極: "high",
};

const defaultProfile: InvestorProfile = {
  style: "穩健",
  risk_tolerance: "medium",
  max_loss_pct: 10,
};

const appTitle = "台股量化預測與新聞情緒之雙軌決策輔助系統";
const sessionTitleMaxLen = 30;

function getLocalUserId() {
  const existing = localStorage.getItem("stock-advisor-user-id");
  if (existing) return existing;
  const created = crypto.randomUUID().replaceAll("-", "");
  localStorage.setItem("stock-advisor-user-id", created);
  return created;
}

function App() {
  const [userId, setUserId] = useState(getLocalUserId);
  const [sessions, setSessions] = useState<ChatSession[]>([
    {
      session_id: "local-session",
      title: "新對話",
      messages: [],
    },
  ]);
  const [activeSessionId, setActiveSessionId] = useState("local-session");
  const [profile, setProfile] = useState<InvestorProfile>(defaultProfile);
  const [apiKey, setApiKey] = useState(sessionStorage.getItem("groq-api-key") ?? "");
  const [model, setModel] = useState(localStorage.getItem("groq-model") ?? "openai/gpt-oss-20b");
  const [draft, setDraft] = useState("");
  const [statusSteps, setStatusSteps] = useState<string[]>([]);
  const [loadError, setLoadError] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(() => !window.matchMedia("(max-width: 900px)").matches);
  const [apiKeyPanelOpen, setApiKeyPanelOpen] = useState(false);
  const [apiKeyVisible, setApiKeyVisible] = useState(false);
  const [profilePanelOpen, setProfilePanelOpen] = useState(false);
  const [followUps, setFollowUps] = useState<string[]>([]);
  const [streamStartedAt, setStreamStartedAt] = useState<number | null>(null);
  const [lastElapsedMs, setLastElapsedMs] = useState<number | null>(null);
  const [clockNow, setClockNow] = useState(() => Date.now());
  const [toast, setToast] = useState<{ id: number; message: string; tone?: "ok" | "warn" } | null>(null);
  const firstStatusTimerRef = useRef<number | null>(null);
  const hasReceivedStatusRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [apiKeyCheck, setApiKeyCheck] = useState<
    { state: "idle" } | { state: "ok"; message: string } | { state: "warn"; message: string }
  >({ state: "idle" });
  const [, startTransition] = useTransition();

  useEffect(() => {
    loadSessions(userId)
      .then((payload) => {
        setLoadError("");
        setUserId(payload.user_id);
        localStorage.setItem("stock-advisor-user-id", payload.user_id);
        setSessions(payload.sessions);
        setActiveSessionId(payload.current_session);
        setProfile(payload.investor_profile);
      })
      .catch(() => {
        const fallbackSession = {
          session_id: "local-session",
          title: "新對話",
          messages: [
            {
              id: "local-welcome",
              role: "assistant" as const,
              content: "目前無法連線到 FastAPI 後端。您仍可輸入訊息，送出時會重試連線。",
              created_at: Date.now() / 1000,
            },
          ],
        };
        setLoadError(`無法連線到 FastAPI 後端（${getApiBaseUrl()}），請確認後端服務或 VITE_API_BASE_URL 設定。`);
        setSessions([fallbackSession]);
        setActiveSessionId(fallbackSession.session_id);
      });
  }, [userId]);

  useEffect(() => {
    sessionStorage.setItem("groq-api-key", apiKey);
  }, [apiKey]);

  // If a valid API key is already present (e.g. restored from sessionStorage),
  // don't keep showing stale "please enter API key" style banners.
  useEffect(() => {
    const trimmed = apiKey.trim();
    if (!trimmed) return;
    if (!/^gsk_[A-Za-z0-9_-]{8,}$/.test(trimmed)) return;
    // Only clear banners that are related to API-key entry/format.
    if (/API Key/.test(loadError)) {
      setLoadError("");
    }
    // Ensure hint reflects that we do have a key stored.
    if (apiKeyCheck.state === "idle") {
      setApiKeyCheck({
        state: "ok",
        message: "API Key 已暫存於本機（本次瀏覽器 session）。送出訊息時會以 X-Groq-API-Key 隨請求送到後端。",
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiKey]);

  useEffect(() => {
    localStorage.setItem("groq-model", model);
  }, [model]);

  useEffect(() => {
    if (!isStreaming) return;
    const timer = window.setInterval(() => setClockNow(Date.now()), 100);
    return () => window.clearInterval(timer);
  }, [isStreaming]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 1800);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    return () => {
      if (firstStatusTimerRef.current !== null) {
        window.clearTimeout(firstStatusTimerRef.current);
      }
      abortControllerRef.current?.abort();
    };
  }, []);

  const activeSession = sessions.find((session) => session.session_id === activeSessionId);
  const messages = activeSession?.messages ?? [];
  const activeTitle = activeSession?.title || appTitle;
  const isConversationStart = !messages.some((message) => message.role === "user");

  function runApiKeyCheck() {
    const trimmed = apiKey.trim();
    if (!trimmed) {
      setApiKeyCheck({ state: "warn", message: "尚未輸入 API Key。請先在左側欄 API Key 面板輸入。" });
      return;
    }
    if (!/^gsk_[A-Za-z0-9_-]{8,}$/.test(trimmed)) {
      setApiKeyCheck({ state: "warn", message: "API Key 格式錯誤，請確認是否貼上完整。" });
      return;
    }
    // A valid key is present; clear any stale "missing/invalid API key" banner text.
    if (/API Key/.test(loadError)) {
      setLoadError("");
    }
    setApiKeyCheck({
      state: "ok",
      message: "API Key 已暫存於本機（本次瀏覽器 session）。送出訊息時會以 X-Groq-API-Key 隨請求送到後端。",
    });
  }

  async function handleNewChat() {
    try {
      const created = await createSession(userId);
      setLoadError("");
      setSessions((current) => [
        ...current,
        { session_id: created.session_id, title: "新對話", messages: created.messages },
      ]);
      setActiveSessionId(created.session_id);
    } catch {
      const sessionId = `local-${crypto.randomUUID()}`;
      // Creating a backend session is nice-to-have; we can still continue locally.
      // Don't block the UI with a red banner for this.
      showToast("無法建立後端對話，已先開啟本機暫存對話。", "warn");
      setSessions((current) => [
        ...current,
        {
          session_id: sessionId,
          title: "新對話",
          messages: [
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: "後端目前無法連線。您可以先輸入訊息，送出時會再次嘗試連線。",
              created_at: Date.now() / 1000,
            },
          ],
        },
      ]);
      setActiveSessionId(sessionId);
    }
  }

  async function handleProfileChange(next: InvestorProfile) {
    const normalized = { ...next, risk_tolerance: riskByStyle[next.style] };
    setProfile(normalized);
    try {
      await updateProfile(userId, normalized);
      setLoadError("");
    } catch {
      setLoadError("偏好已暫存在前端，但尚未同步到 FastAPI 後端。");
    }
  }

  async function handleSubmit(event?: FormEvent, override?: string) {
    event?.preventDefault();
    const content = (override ?? draft).trim();
    if (!content || isStreaming) return;
    const trimmedKey = apiKey.trim();
    if (!trimmedKey) {
      setLoadError("尚未輸入 API Key，請先在左側欄 API Key 面板輸入。");
      setApiKeyPanelOpen(true);
      return;
    }
    if (!/^gsk_[A-Za-z0-9_-]{8,}$/.test(trimmedKey)) {
      setLoadError("API Key 格式錯誤，請確認是否貼上完整。");
      setApiKeyPanelOpen(true);
      setApiKeyCheck({ state: "warn", message: "API Key 格式錯誤，請確認是否貼上完整。" });
      return;
    }
    setFollowUps([]);
    setLastElapsedMs(null);
    setStatusSteps(["準備分析..."]);
    hasReceivedStatusRef.current = false;
    if (firstStatusTimerRef.current !== null) {
      window.clearTimeout(firstStatusTimerRef.current);
    }
    firstStatusTimerRef.current = window.setTimeout(() => {
      if (!hasReceivedStatusRef.current) {
        setStatusSteps(["後端仍在啟動或分析中，請稍候..."]);
      }
    }, 8000);
    const requestStarted = Date.now();
    setStreamStartedAt(requestStarted);
    setClockNow(requestStarted);
    const abortController = new AbortController();
    abortControllerRef.current = abortController;


    let targetSessionId = activeSessionId;
    if (!targetSessionId) {
      targetSessionId = `local-${crypto.randomUUID()}`;
      setSessions((current) => [
        ...current,
        {
          session_id: targetSessionId,
          title: "新對話",
          messages: [],
        },
      ]);
      setActiveSessionId(targetSessionId);
    }
    setDraft("");
    setIsStreaming(true);
    setLoadError("");
    setApiKeyPanelOpen(false);
    setProfilePanelOpen(false);

    const optimisticUser: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      created_at: Date.now() / 1000,
    };
    let liveAssistant: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      created_at: Date.now() / 1000,
    };

    setSessions((current) =>
      current.map((session) =>
        session.session_id === targetSessionId
          ? {
              ...session,
              title: session.title === "新對話" ? content.slice(0, sessionTitleMaxLen) : session.title,
              messages: [...session.messages, optimisticUser, liveAssistant],
            }
          : session,
      ),
    );

    if (targetSessionId.startsWith("local")) {
      try {
        const created = await createSession(userId);
        const previousLocalId = targetSessionId;
        targetSessionId = created.session_id;
        setActiveSessionId(created.session_id);
        setSessions((current) =>
          current.map((session) =>
            session.session_id === previousLocalId
              ? {
                  ...session,
                  session_id: created.session_id,
                }
              : session,
          ),
        );
      } catch {
        targetSessionId = activeSessionId || targetSessionId;
      }
    }

    try {
      await streamMessage(
        targetSessionId,
        { user_id: userId, content, profile, model },
        apiKey,
        (streamEvent: StreamEvent) => {
          startTransition(() => {
            if (streamEvent.event === "message") {
              setSessions((current) =>
                current.map((session) => {
                  if (session.session_id !== targetSessionId) return session;
                  const next = session.messages.filter((message) => message.id !== optimisticUser.id);
                  const assistantIndex = next.findIndex((message) => message.id === liveAssistant.id);
                  if (assistantIndex >= 0) {
                    next.splice(assistantIndex, 0, streamEvent.data.message);
                  } else {
                    next.push(streamEvent.data.message, liveAssistant);
                  }
                  return { ...session, messages: next };
                }),
              );
            }
            if (streamEvent.event === "status") {
              hasReceivedStatusRef.current = true;
              if (firstStatusTimerRef.current !== null) {
                window.clearTimeout(firstStatusTimerRef.current);
                firstStatusTimerRef.current = null;
              }
              setStatusSteps(prev => {
                const text = cleanStatusText(streamEvent.data.text);
                const tag = text.match(/^\[\d+\/\d+\]/);
                if (tag && prev.length > 0 && prev[prev.length - 1].startsWith(tag[0])) {
                  return [...prev.slice(0, -1), text];
                }
                return [...prev, text];
              });
            }
            if (streamEvent.event === "token") {
              liveAssistant = { ...liveAssistant, content: liveAssistant.content + streamEvent.data.text };
              setSessions((current) =>
                current.map((session) => {
                  if (session.session_id !== targetSessionId) return session;
                  const next = session.messages.map((message) =>
                    message.id === liveAssistant.id ? liveAssistant : message,
                  );
                  return { ...session, messages: next };
                }),
              );
            }
            if (streamEvent.event === "dashboard") {
              liveAssistant = { ...liveAssistant, dashboard_data: streamEvent.data.dashboard_data };
              setSessions((current) =>
                current.map((session) => {
                  if (session.session_id !== targetSessionId) return session;
                  return {
                    ...session,
                    messages: session.messages.map((message) =>
                      message.id === liveAssistant.id ? liveAssistant : message,
                    ),
                  };
                }),
              );
            }
            if (streamEvent.event === "done") {
              const completedMs = Date.now() - requestStarted;
              setSessions((current) => {
                const updated = current.map((session) => {
                  if (session.session_id !== targetSessionId) return session;
                  const next = session.messages.map((message) =>
                    message.id === liveAssistant.id ? streamEvent.data.message : message,
                  );
                  return { ...session, messages: next };
                });
                const sess = updated.find(s => s.session_id === targetSessionId);
                if (sess) setFollowUps(buildFollowUps(sess.messages));
                return updated;
              });
              setIsStreaming(false);
              setStatusSteps([]);
              setStreamStartedAt(null);
              setLastElapsedMs(completedMs);
              abortControllerRef.current = null;
              if (firstStatusTimerRef.current !== null) {
                window.clearTimeout(firstStatusTimerRef.current);
                firstStatusTimerRef.current = null;
              }
            }
            if (streamEvent.event === "error") {
              const completedMs = Date.now() - requestStarted;
              setIsStreaming(false);
              liveAssistant = { ...liveAssistant, content: streamEvent.data.message };
              setFollowUps([]);
              setSessions((current) =>
                current.map((session) => {
                  if (session.session_id !== targetSessionId) return session;
                  return {
                    ...session,
                    messages: session.messages.map((message) =>
                      message.id === liveAssistant.id ? liveAssistant : message,
                    ),
                  };
                }),
              );
              setStatusSteps([]);
              setStreamStartedAt(null);
              setLastElapsedMs(completedMs);
              abortControllerRef.current = null;
              if (firstStatusTimerRef.current !== null) {
                window.clearTimeout(firstStatusTimerRef.current);
                firstStatusTimerRef.current = null;
              }
            }
          });
        },
        abortController.signal,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      const message = error instanceof Error ? error.message : "Unable to stream message";
      liveAssistant = { ...liveAssistant, content: `送出失敗：${message}` };
      setLoadError(`訊息未送達，請確認 FastAPI 後端（${getApiBaseUrl()}）與 API key 設定後再試一次。`);
      setStatusSteps([]);
      setStreamStartedAt(null);
      setLastElapsedMs(Date.now() - requestStarted);
      abortControllerRef.current = null;
      if (firstStatusTimerRef.current !== null) {
        window.clearTimeout(firstStatusTimerRef.current);
        firstStatusTimerRef.current = null;
      }
      setSessions((current) =>
        current.map((session) => {
          if (session.session_id !== targetSessionId) return session;
          return {
            ...session,
            messages: session.messages.map((item) => (item.id === liveAssistant.id ? liveAssistant : item)),
          };
        }),
      );
    } finally {
      if (firstStatusTimerRef.current !== null) {
        window.clearTimeout(firstStatusTimerRef.current);
        firstStatusTimerRef.current = null;
      }
      setIsStreaming(false);
    }
  }

  function handlePauseStreaming() {
    if (!isStreaming) return;
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    const elapsedMs = streamStartedAt ? Date.now() - streamStartedAt : 0;
    setSessions((current) =>
      current.map((session) => {
        if (session.session_id !== activeSessionId) return session;
        const next = [...session.messages];
        for (let index = next.length - 1; index >= 0; index -= 1) {
          const message = next[index];
          if (message.role === "assistant" && !message.content.trim() && !message.dashboard_data) {
            next[index] = { ...message, content: "已暫停分析。" };
            break;
          }
        }
        return { ...session, messages: next };
      }),
    );
    setLastElapsedMs(elapsedMs);
    setStatusSteps(["已暫停"]);
    setStreamStartedAt(null);
    setIsStreaming(false);
  }

  function handleComposerKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Ctrl+Enter (or Cmd+Enter on Mac) submits; plain Enter adds newline
    if (event.key !== "Enter" || !(event.ctrlKey || event.metaKey) || event.nativeEvent.isComposing) return;
    event.preventDefault();
    void handleSubmit();
  }

  function normalizeTicker(value: string) {
    return value.trim().toUpperCase();
  }

  function showToast(message: string, tone: "ok" | "warn" = "ok") {
    setToast({ id: Date.now(), message, tone });
  }

  // followUps is now state, not derived

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <button
          className="sidebar-backdrop"
          onClick={() => {
            setSidebarOpen(false);
            setApiKeyPanelOpen(false);
            setProfilePanelOpen(false);
          }}
          aria-label="關閉側欄"
        />
      )}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-top">
          <div className="workspace-mark">{appTitle}</div>
          <button
            className="icon-button"
            onClick={() => {
              setSidebarOpen(false);
              setApiKeyPanelOpen(false);
              setProfilePanelOpen(false);
            }}
            title="收合側欄"
          >
            <PanelLeftClose size={21} />
          </button>
        </div>
        <div className="sidebar-nav" aria-label="主要導覽">
          <button className="nav-item" onClick={handleNewChat}>
            <MessageSquarePlus size={17} />
            新聊天
          </button>
        </div>
        <div className="sidebar-section-title">聊天</div>
        <div className="session-list">
          {sessions
            .slice()
            .reverse()
            .map((session) => (
              <div
                className={`session-item ${session.session_id === activeSessionId ? "active" : ""}`}
                key={session.session_id}
              >
                <button className="session-link" onClick={() => setActiveSessionId(session.session_id)}>
                  <span>{session.title}</span>
                </button>
              </div>
            ))}
        </div>
        <div className="sidebar-bottom">
          <button
            className="settings-link"
            onClick={() => {
              setProfilePanelOpen(false);
              setApiKeyPanelOpen((open) => !open);
            }}
          >
            <Settings2 size={18} />
            API Key
          </button>
          {apiKeyPanelOpen && (
            <section className="sidebar-api-panel">
              <label>
                <span>Groq API Key</span>
                <div className="input-with-icon">
                  <KeyRound size={17} />
                  <input
                    value={apiKey}
                    onChange={(event) => {
                      setApiKey(event.target.value);
                      // Clear stale banner once user starts editing.
                      if (loadError) setLoadError("");
                      setApiKeyCheck({ state: "idle" });
                    }}
                    onBlur={runApiKeyCheck}
                    type={apiKeyVisible ? "text" : "password"}
                    placeholder="gsk_..."
                  />
                  <button
                    type="button"
                    className="input-visibility-toggle"
                    onClick={() => setApiKeyVisible((visible) => !visible)}
                    title={apiKeyVisible ? "隱藏 API Key" : "顯示 API Key"}
                  >
                    {apiKeyVisible ? <Eye size={16} /> : <EyeOff size={16} />}
                  </button>
                </div>
              </label>
              {apiKeyCheck.state !== "idle" && (
                <div className={`api-key-hint ${apiKeyCheck.state === "ok" ? "ok" : "warn"}`}>
                  {apiKeyCheck.message}
                </div>
              )}
            </section>
          )}
        </div>
      </aside>

      <main className="chat-pane">
        <header className="topbar">
          <div className="topbar-inner">
            {!sidebarOpen && (
              <button
                className="icon-button"
                onClick={() => {
                  setApiKeyPanelOpen(false);
                  setProfilePanelOpen(false);
                  setSidebarOpen(true);
                }}
                title="展開側欄"
              >
                <Menu size={22} />
              </button>
            )}
            <div className="chat-title">
              <span>{activeTitle}</span>
            </div>
          </div>
        </header>

        <section className="messages">
          {loadError && (
            <div className="error-banner">
              <CircleAlert size={17} />
              <div className="error-banner-body">
                <div className="error-banner-text">{loadError}</div>
                <div className="error-actions">
                  <button type="button" className="error-action" onClick={runApiKeyCheck}>
                    Check API Key
                  </button>
                </div>
                {apiKeyCheck.state !== "idle" && (
                  <div className={`error-diagnostics ${apiKeyCheck.state === "ok" ? "ok" : "warn"}`}>
                    {apiKeyCheck.message}
                  </div>
                )}
              </div>
            </div>
          )}
          {isConversationStart && (
            <div className="empty-state">
              <h1>{appTitle}</h1>
              <p>您可以在下方詢問台股投資相關問題，系統會自動為您分析。</p>
            </div>
          )}
          {messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              onCopyFeedback={showToast}
              canCopy={message.role === "assistant" && messages.slice(0, index).some((item) => item.role === "user")}
            />
          ))}
          {statusSteps.length > 0 && (
            <div className="status-row">
              <Bot size={16} />
              <span>{statusSteps[statusSteps.length - 1]}</span>
              <span className="status-timer">
                {streamStartedAt ? formatElapsed(clockNow - streamStartedAt) : lastElapsedMs !== null ? `總耗時 ${formatElapsed(lastElapsedMs)}` : ""}
              </span>
              {isStreaming && (
                <button type="button" className="pause-stream-button" onClick={handlePauseStreaming}>
                  <Square size={14} />
                  暫停
                </button>
              )}
            </div>
          )}
          {!isStreaming && statusSteps.length === 0 && lastElapsedMs !== null && (
            <div className="status-row status-row-done">
              <Bot size={16} />
              <span>已完成</span>
              <span className="status-timer">總耗時 {formatElapsed(lastElapsedMs)}</span>
            </div>
          )}
        </section>

        <section className="composer-wrap">
          {!!followUps.length && (
            <div className="followups">
              {followUps.map((question, index) => (
                <button
                  key={question}
                  onClick={(event) => handleSubmit(event, question)}
                  disabled={isStreaming}
                >
                  {question}
                </button>
              ))}
            </div>
          )}
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="請在此輸入欲分析之問題，開始建立一段新的對話。"
              rows={1}
              data-testid="chat-composer"
            />
            <div className="composer-controls">
              <div className="composer-tools" style={{ position: "relative" }}>
                {profilePanelOpen && (
                  <section className="profile-panel">
                    <div className="settings-heading">
                      <strong>風險設定</strong>
                      <span>
                        {profile.style} / 最大虧損 {profile.max_loss_pct}%
                      </span>
                    </div>
                    <label>
                      <span>投資風格</span>
                      <select
                        value={profile.style}
                        onChange={(event) =>
                          handleProfileChange({ ...profile, style: event.target.value as InvestorProfile["style"] })
                        }
                      >
                        <option>保守</option>
                        <option>穩健</option>
                        <option>積極</option>
                      </select>
                    </label>
                    <label>
                      <span>最大可接受虧損</span>
                      <select
                        value={profile.max_loss_pct}
                        onChange={(event) =>
                          handleProfileChange({ ...profile, max_loss_pct: Number(event.target.value) })
                        }
                      >
                        {[5, 10, 15, 20, 30].map((value) => (
                          <option key={value} value={value}>
                            {value}%
                          </option>
                        ))}
                      </select>
                    </label>
                  </section>
                )}
                <button
                  type="button"
                  className="ghost-tool"
                  onClick={() => {
                    setApiKeyPanelOpen(false);
                    setProfilePanelOpen((open) => !open);
                  }}
                  title="風險設定"
                >
                  <SlidersHorizontal size={16} />
                </button>
              </div>
              <button className="send-button" disabled={isStreaming || !draft.trim()} title="送出">
                <Send size={19} />
              </button>
            </div>
          </form>
          <div className="fine-print">金融市場有不確定性，本分析僅供參考。</div>
        </section>
        {toast && (
          <div className={`app-toast ${toast.tone ?? "ok"}`} key={toast.id} role="status" aria-live="polite">
            {toast.message}
          </div>
        )}
      </main>
    </div>
  );
}

function MessageBubble({
  message,
  onCopyFeedback,
  canCopy,
}: {
  message: ChatMessage;
  onCopyFeedback: (message: string, tone?: "ok" | "warn") => void;
  canCopy: boolean;
}) {
  const normalized = String(message.content ?? "")
    .replace(/<br\s*\/?\s*>/gi, "\n")
    .replace(/^已進入.*$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  const shouldShowCopy = canCopy && normalized.length > 0;
  return (
    <article className={`message ${message.role}`}>
      <div className="message-body">
        {message.dashboard_data && <Dashboard dashboard={message.dashboard_data} />}
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{normalized}</ReactMarkdown>
        {shouldShowCopy && (
          <div className="message-footer">
            <button
              type="button"
              className="copy-summary-button"
              onClick={() => {
                void copyMessageContent(normalized)
                  .then(() => onCopyFeedback("已複製", "ok"))
                  .catch(() => onCopyFeedback("複製失敗，請再試一次", "warn"));
              }}
              title="複製整段回覆"
            >
              <Copy size={14} />
              複製回覆
            </button>
          </div>
        )}
      </div>
    </article>
  );
}

function Dashboard({ dashboard }: { dashboard: DashboardData }) {
  const [expandNews, setExpandNews] = useState(false);
  const [expandQuant, setExpandQuant] = useState(false);
  const [chartRange, setChartRange] = useState<"1M" | "3M" | "1Y">("3M");
  const quant = dashboard.quant_data ?? {};
  const news = dashboard.news_data ?? {};
  const stock = dashboard.stock_data ?? {};
  const allChartData = (dashboard.chart_data ?? []).map((d: Record<string, unknown>) => ({
    date: String(d.date ?? "").slice(0, 10),
    close: Number(d.Close ?? 0),
    open: Number(d.Open ?? 0),
    high: Number(d.High ?? 0),
    low: Number(d.Low ?? 0),
  }));

  // 1M≈21 trading days, 3M≈63, 1Y≈252
  const rangeMap = { "1M": 21, "3M": 63, "1Y": 252 };
  const chartData = allChartData.slice(-rangeMap[chartRange]);

  const rawNews = asRecordArray(news.raw_news);
  const modelDetails = asRecordArray(quant.model_details);
  const predictedReturn = toPercent(quant.predicted_return, 3);
  const expectedMove = toPriceMove(stock.latest_price, quant.predicted_return);
  const sentimentScore = formatValue(news.sentiment_score ?? 0);
  const trust = evaluateTrust(quant, news);
  const consistency = evaluateConsistency(quant, news);
  const displayName =
    typeof dashboard.company_name === "string" && /^\d{4,6}$/.test(dashboard.company_name.trim())
      ? formatValue(stock.company_name ?? dashboard.company_name)
      : dashboard.company_name;
  const quantRows = [
    ["預測報酬率", predictedReturn],
    ["綜合訊號", formatValue(quant.signal ?? "未知")],
    ["市場狀態", formatValue(quant.regime ?? "未知")],
    ["歷史最大回撤", formatValue(quant.max_dd ?? "未知")],
    ["最佳模型名稱", formatValue(quant.model_name ?? "未知")],
  ];

  return (
    <section className="dashboard">
      <div className="dashboard-title">
        <div>
          <span>{displayName}</span>
          <small>投資人設定：{dashboard.investor_profile?.style ?? "穩健"} / 最大可接受虧損 {dashboard.investor_profile?.max_loss_pct ?? 10}%</small>
        </div>
        <strong>{dashboard.ticker}</strong>
      </div>
      <div className="metric-grid">
        <Metric label="目前價格" value={typeof stock.latest_price === "number" ? stock.latest_price.toFixed(2) : String(stock.latest_price ?? "未知")} />
        <Metric label="量化訊號" value={String(quant.signal ?? "未知")} />
        <Metric label="預期報酬" value={expectedMove === "未知" ? predictedReturn : `${predictedReturn} / ${expectedMove}`} />
        <Metric label="情緒分數" value={sentimentScore} />
        <Metric label="市場狀態" value={String(quant.regime ?? "未知")} />
        <Metric label="近一年 Max DD" value={String(quant.max_dd ?? "未知")} />
        <Metric label="建議可信度" value={trust.level} />
        <Metric label="新聞數量" value={formatNewsCount(rawNews.length, news.news_count_status)} />
      </div>
      {consistency && (
        <div className={`consistency-card ${consistency.severity}`}>
          <div className="consistency-title">{consistency.title}</div>
          <div className="consistency-body">{consistency.summary}</div>
          <div className="consistency-actions">
            {consistency.nextSteps.map((step) => (
              <div className="consistency-step" key={step}>
                {step}
              </div>
            ))}
          </div>
        </div>
      )}
      {!consistency && trust.reason && <div className="trust-note">{trust.reason}</div>}
      {!!chartData.length && (
        <div className="mini-chart">
          <div className="chart-range-controls">
            {(["1M", "3M", "1Y"] as const).map(r => (
              <button
                key={r}
                className={`chart-range-btn ${chartRange === r ? "active" : ""}`}
                onClick={() => setChartRange(r)}
              >{r}</button>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData} margin={{ top: 4, right: 20, bottom: 10, left: 0 }}>
              <XAxis
                dataKey="date"
                tick={{ fontSize: 10, fill: "#999" }}
                tickFormatter={(v: string) => String(v).slice(5, 10)}
                interval="preserveStartEnd"
                padding={{ left: 8, right: 24 }}
                tickMargin={8}
                height={28}
                minTickGap={18}
              />
              <YAxis
                domain={["auto", "auto"]}
                tick={{ fontSize: 10, fill: "#999" }}
                tickFormatter={(v: number) => v.toFixed(0)}
                width={44}
              />
              <Tooltip
                formatter={(value: unknown) => [Number(value).toFixed(2), "收盤價"]}
                labelFormatter={(label: any) => `日期：${String(label ?? "")}`}
              />
              <Line type="monotone" dataKey="close" stroke="var(--accent)" strokeWidth={2} dot={false} activeDot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="dashboard-section">
        <button className="section-toggle" onClick={() => setExpandNews(n => !n)}>
          <span>情緒新聞清單</span>
          <strong>{formatNewsCount(rawNews.length, news.news_count_status)} {expandNews ? "▲" : "▼"}</strong>
        </button>
        {expandNews && (
          rawNews.length ? (
            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>新聞標題</th>
                    <th>FinBERT分數</th>
                    <th>情緒標籤</th>
                  </tr>
                </thead>
                <tbody>
                  {rawNews.map((item, index) => (
                    <tr key={`${formatValue(item["新聞標題"])}-${index}`}>
                      <td>{formatValue(item["新聞標題"] ?? item.title ?? "未知")}</td>
                      <td>{formatValue(item["FinBERT分數"] ?? item.score ?? "")}</td>
                      <td>{formatValue(item["情緒標籤"] ?? item.label ?? "無")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted-line">過去 5 天內未能搜尋到相關新聞。</p>
          )
        )}
      </div>
      <div className="dashboard-section">
        <button className="section-toggle" onClick={() => setExpandQuant(q => !q)}>
          <span>量化模型參數</span>
          <strong>{formatValue(quant.model_name ?? "模型明細")} {expandQuant ? "▲" : "▼"}</strong>
        </button>
        {expandQuant && (
          <div className="data-table-wrap compact">
            <table className="data-table">
              <tbody>
                {quantRows.map(([label, value]) => (
                  <tr key={label}>
                    <th>{label}</th>
                    <td>{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {expandQuant && !!modelDetails.length && (
          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  {Object.keys(modelDetails[0]).map((key) => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {modelDetails.map((row, index) => (
                  <tr key={index}>
                    {Object.keys(modelDetails[0]).map((key) => (
                      <td key={key}>{formatValue(row[key])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

const metricTooltips: Record<string, string> = {
  "目前價格": "目前的收盤價。",
  "近一年 Max DD": "過去一年內可能會遇到的最大虧損幅度，衡量下檔風險。",
  "市場狀態": "模型依據波動度計算出的目前市場趨勢。",
  "預期報酬": "顯示預期報酬的百分比與轉換後的價差。",
  "量化訊號": "基於數據推算出的交易訊號，建議搭配風險控管使用。",
  "情緒分數": "基於LLM過濾與Hybrid FinBERT計算出的分數，正數偏多、負數偏空。",
  "建議可信度": "衡量量化訊號與新聞情緒是否同向，所給出的信心水準。",
  "新聞數量": "過去五天內抓取到並過濾掉雜訊後的純淨新聞篇數。"
};

function Metric({ label, value }: { label: string; value: string }) {
  const tooltipText = metricTooltips[label];

  return (
    <div className={`metric ${tooltipText ? "has-tooltip" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {tooltipText && <div className="metric-tooltip">{tooltipText}</div>}
    </div>
  );
}

function buildFollowUps(messages: ChatMessage[]) {
  const hasUserMessage = messages.some((m) => m.role === "user" && m.content.trim());
  if (!hasUserMessage) {
    return [];
  }
  const STOCK_POOL: Array<{ name: string; ticker: string }> = [
    { name: "台積電", ticker: "2330" },
    { name: "聯發科", ticker: "2454" },
    { name: "鴻海", ticker: "2317" },
    { name: "國巨", ticker: "2327" },
    { name: "長榮", ticker: "2603" },
    { name: "陽明", ticker: "2609" },
    { name: "世界", ticker: "5347" },
    { name: "廣達", ticker: "2382" },
    { name: "緯創", ticker: "3231" },
    { name: "華邦電", ticker: "2344" },
    { name: "金寶", ticker: "2312" },
    { name: "立積", ticker: "4968" },
  ];

  const pick = (exclude: Set<string> = new Set()) => {
    const candidates = STOCK_POOL.filter((s) => !exclude.has(s.ticker));
    const chosen = candidates[Math.floor(Math.random() * candidates.length)] || STOCK_POOL[0];
    return chosen;
  };

  const a = pick();
  const b = pick(new Set([a.ticker]));
  const c = pick(new Set([a.ticker, b.ticker]));
  const A = `${a.name}（${a.ticker}）`;
  const B = `${b.name}（${b.ticker}）`;
  const C = `${c.name}（${c.ticker}）`;

  const templates = [
    () => `請用「一句話結論 / 主要理由 / 可執行操作」三段回答：${A} 目前適合分批進場嗎？`,
    () => `請給我可執行的停損規則（價位/條件/例外）：${A} 如果跌破月線，停損可以怎麼設？`,
    () => `請用白話解讀，並點出最重要的 2 個風險：${A} 的量化訊號和回撤（Max DD）該怎麼看？`,
    () => `請直接選 1 檔並說明原因：${A} 和 ${B} 如果只能選一檔，現階段我該優先哪一檔？`,
    () => `請列出 2 個不適合追高的理由：${A}、${B}、${C} 當中哪一檔最不適合追高？`,
    () => `請給我一個保守配置比例（用百分比）：${A} 如果大盤轉弱，部位該怎麼分配？`,
  ];

  const results: string[] = [];
  const used = new Set<number>();
  while (results.length < 3 && used.size < templates.length) {
    const idx = Math.floor(Math.random() * templates.length);
    if (used.has(idx)) continue;
    used.add(idx);
    results.push(templates[idx]());
  }
  return results;
}

async function copyMessageContent(content: string) {
  const fallback = content.trim();
  if (!navigator.clipboard?.writeText) {
    throw new Error("clipboard-unavailable");
  }
  await navigator.clipboard.writeText(fallback);
}

function formatElapsed(ms: number) {
  const totalSeconds = Math.max(0, ms) / 1000;
  return `${totalSeconds.toFixed(1)}s`;
}

function cleanStatusText(text: unknown) {
  return String(text ?? "")
    .replace(/\s*\([^)]*(?:總耗時|耗時|[0-9.]+s)[^)]*\)/g, "")
    .replace(/\s*總耗時\s*[0-9.]+s/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function asRecordArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => isRecord(item)) : [];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "無";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

function toPercent(value: unknown, digits = 2) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "未知";
  return `${(numeric * 100).toFixed(digits)}%`;
}

function toPriceMove(price: unknown, predictedReturn: unknown) {
  const numericPrice = Number(price);
  const numericReturn = Number(predictedReturn);
  if (!Number.isFinite(numericPrice) || !Number.isFinite(numericReturn)) return "未知";
  return `${(numericPrice * numericReturn).toFixed(2)}元`;
}

function formatNewsCount(count: number, status: unknown) {
  if (count > 0) return `共 ${count} 則`;
  const normalized = String(status ?? "");
  if (normalized === "fetch_failed") return "抓取失敗";
  if (normalized === "few") return "偏少";
  if (normalized === "none") return "無新聞";
  return "未知";
}

function evaluateTrust(quant: any, news: any) {
  const signal = String(quant?.signal ?? "觀望");
  const sentimentScore = Number(news?.sentiment_score ?? 0);
  const newsStatus = String(news?.news_count_status ?? "sufficient");
  
  let maxDdStr = String(quant?.max_dd ?? "0%");
  const maxDd = Number(maxDdStr.replace(/[()%]/g, "")) / 100;

  const bullish = ["買入", "強烈買入", "反彈契機"].some((keyword) => signal.includes(keyword));
  const bearish = ["賣出", "強烈賣出", "持倉", "強烈觀望"].some((keyword) => signal.includes(keyword));
  const bullishSentiment = sentimentScore > 0.15;
  const bearishSentiment = sentimentScore < -0.05;

  const aligned = (bullish && bullishSentiment) || (bearish && bearishSentiment);
  const conflicted = (bullish && bearishSentiment) || (bearish && bullishSentiment);
  const weakSignal = ["觀望", "資料過少", "無法預測", "集成失敗", "運算錯誤"].includes(signal);
  const dataLimited = ["none", "few", "fetch_failed"].includes(newsStatus);
  const highDrawdown = maxDd <= -0.2;

  let level = "中";
  let reason = "量化或情緒僅單邊支持，建議搭配風險控管。";

  if (aligned && !dataLimited && !highDrawdown) {
    level = "高";
    reason = "量化與情緒同向，且新聞樣本足夠。";
  } else if (conflicted || dataLimited || highDrawdown) {
    level = "低";
    if (conflicted) reason = "量化訊號與市場情緒分歧。";
    else if (highDrawdown) reason = "近一年回撤偏大，波動風險較高。";
    else if (newsStatus === "fetch_failed") reason = "新聞抓取失敗，情緒分數暫時不具代表性。";
    else reason = "新聞樣本不足，情緒代表性較弱。";
  } else if (weakSignal) {
    level = "中";
    reason = "訊號偏保守或資料條件普通。";
  }

  if (bullish && sentimentScore < -0.2) {
    reason = "⚠️ 信心度警告：量化訊號偏多，但近期市場情緒顯著偏空，建議保守評估。";
  } else if (bearish && sentimentScore > 0.2) {
    reason = "⚠️ 信心度警告：量化訊號偏空或遇壓，但近期市場情緒火熱，需留意軋空風險。";
  }

  return { level, reason };
}

function evaluateConsistency(quant: any, news: any): null | {
  severity: "warn" | "info";
  title: string;
  summary: string;
  nextSteps: string[];
} {
  const signal = String(quant?.signal ?? "觀望");
  const sentimentScore = Number(news?.sentiment_score ?? 0);
  const newsStatus = String(news?.news_count_status ?? "sufficient");
  const bullish = ["買入", "強烈買入", "反彈契機"].some((keyword) => signal.includes(keyword));
  const bearish = ["賣出", "強烈賣出", "持倉", "強烈觀望"].some((keyword) => signal.includes(keyword));
  const bullishSentiment = sentimentScore > 0.15;
  const bearishSentiment = sentimentScore < -0.05;
  const aligned = (bullish && bullishSentiment) || (bearish && bearishSentiment);
  const conflicted = (bullish && bearishSentiment) || (bearish && bullishSentiment);

  const dataLimited = ["none", "few", "fetch_failed"].includes(newsStatus);
  const maxDdStr = String(quant?.max_dd ?? "0%");
  const maxDd = Number(maxDdStr.replace(/[()%]/g, "")) / 100;
  const highDrawdown = Number.isFinite(maxDd) && maxDd <= -0.2;

  if (aligned && !dataLimited && !highDrawdown) return null;

  const nextSteps: string[] = [];
  if (conflicted) {
    nextSteps.push("量化與情緒分歧時，先降低倉位或採分批進出，等待訊號一致再加碼。");
    nextSteps.push("回看最近 5 天重大新聞/事件，確認情緒分數是否被單一事件帶偏。");
  }
  if (dataLimited) {
    nextSteps.push(newsStatus === "fetch_failed" ? "新聞抓取失敗，建議先以量化與價格風險為主，稍後重跑新聞分析。" : "新聞樣本偏少，情緒訊號僅供輔助參考，建議避免過度依賴。");
  }
  if (highDrawdown) {
    nextSteps.push("近一年回撤偏大，建議更嚴格的停損與風險上限（配合你的最大可接受虧損）。");
  }

  if (!nextSteps.length) {
    nextSteps.push("訊號尚不明確，建議以風險控管為主，並觀察後續資料更新。");
  }

  const title = conflicted ? "一致性檢查：訊號分歧" : "一致性檢查：風險提醒";
  const summary = conflicted
    ? "量化訊號與市場情緒並未同向，短線決策不確定性較高。"
    : "資料或波動風險條件偏弱，建議先用風險控管做保護。";

  return { severity: conflicted ? "warn" : "info", title, summary, nextSteps };
}

createRoot(document.getElementById("root")!).render(<App />);
