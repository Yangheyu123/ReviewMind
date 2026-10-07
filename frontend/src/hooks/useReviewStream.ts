import { useEffect, useRef, useCallback } from 'react';
import { useReviewStore } from '../store/reviewStore';
import type {
  ProgressEvent,
  ChunkEvent,
  FindingEvent,
  Finding,
  WarningEvent,
  DoneEvent,
  ErrorEvent,
} from '../types';

/** SSE 连接超时（毫秒）：超过此时间未收到任何事件则判定为连接异常 */
const SSE_CONNECT_TIMEOUT_MS = 15_000;
/** SSE 静默超时（毫秒）：超过此时间未收到任何数据则判定为连接中断 */
const SSE_IDLE_TIMEOUT_MS = 45_000;
/** 最大重连次数 */
const MAX_RECONNECT_ATTEMPTS = 3;
/** 重连基础延迟（毫秒） */
const RECONNECT_BASE_DELAY_MS = 2000;

export function useReviewStream(jobId: string | undefined) {
  const store = useReviewStore();
  const timerRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);
  const connectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectCountRef = useRef(0);
  const mountedRef = useRef(true);

  /** 重置空闲计时器 */
  const resetIdleTimer = useCallback(() => {
    if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    idleTimerRef.current = setTimeout(() => {
      if (!mountedRef.current) return;
      console.warn('[SSE] 连接静默超时，无数据返回');
      store.setSSEStatus('error');
      eventSourceRef.current?.close();
    }, SSE_IDLE_TIMEOUT_MS);
  }, [store]);

  /** 清除所有计时器 */
  const clearAllTimers = useCallback(() => {
    if (connectTimerRef.current) {
      clearTimeout(connectTimerRef.current);
      connectTimerRef.current = null;
    }
    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }
    timerRef.current.forEach(clearTimeout);
    timerRef.current = [];
  }, []);

  /** 建立 SSE 连接 */
  const connectSSE = useCallback(() => {
    if (!jobId || !mountedRef.current) return;

    const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1';
    const streamUrl = `${apiBaseUrl}/review/stream/${jobId}`;

    // 关闭旧连接
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const eventSource = new EventSource(streamUrl);
    eventSourceRef.current = eventSource;

    // 连接超时计时器
    connectTimerRef.current = setTimeout(() => {
      if (eventSource.readyState === EventSource.CONNECTING) {
        console.warn('[SSE] 连接超时，无法建立连接');
        store.setSSEStatus('error');
        eventSource.close();
      }
    }, SSE_CONNECT_TIMEOUT_MS);

    // 连接建立
    eventSource.onopen = () => {
      if (!mountedRef.current) return;
      if (connectTimerRef.current) {
        clearTimeout(connectTimerRef.current);
        connectTimerRef.current = null;
      }
      store.setSSEStatus('connected');
      reconnectCountRef.current = 0; // 重置重连计数
      resetIdleTimer();
    };

    // 通用消息监听（兼容没有明确 event: 字段的消息）
    eventSource.onmessage = (_e: MessageEvent) => {
      if (!mountedRef.current) return;
      resetIdleTimer();
      // 尝试解析为各类事件
      try {
        const data = JSON.parse(_e.data);
        if (data && typeof data === 'object') {
          dispatchSSEEvent('progress', data);
        }
      } catch {
        // 非 JSON 数据，忽略
      }
    };

    // 按事件类型监听
    eventSource.addEventListener('progress', (e: MessageEvent) => {
      if (!mountedRef.current) return;
      resetIdleTimer();
      try {
        const data: ProgressEvent = JSON.parse(e.data);
        store.setProgress(data);
      } catch (err) {
        console.warn('[SSE] 解析 progress 事件失败:', err);
      }
    });

    eventSource.addEventListener('finding', (e: MessageEvent) => {
      if (!mountedRef.current) return;
      resetIdleTimer();
      try {
        const data: FindingEvent = JSON.parse(e.data);
        store.addFinding(data as unknown as Finding);
      } catch (err) {
        console.warn('[SSE] 解析 finding 事件失败:', err);
      }
    });

    eventSource.addEventListener('chunk', (e: MessageEvent) => {
      if (!mountedRef.current) return;
      resetIdleTimer();
      try {
        const data: ChunkEvent = JSON.parse(e.data);
        if (data.target === 'summary') {
          store.addSummaryChunk(data.content);
        } else if (data.target === 'files') {
          // 文件列表 JSON 数组字符串
          try {
            const files = JSON.parse(data.content);
            if (Array.isArray(files)) {
              store.setFileList(files);
            }
          } catch {
            store.setFileList([data.content]);
          }
        } else {
          store.addReportChunk(data.content);
        }
      } catch (err) {
        console.warn('[SSE] 解析 chunk 事件失败:', err);
      }
    });

    eventSource.addEventListener('heartbeat', () => {
      // 后端 LLM 阶段可能长时间无业务事件，命名心跳用于重置静默计时器
      if (!mountedRef.current) return;
      resetIdleTimer();
    });

    eventSource.addEventListener('warning', (e: MessageEvent) => {
      if (!mountedRef.current) return;
      resetIdleTimer();
      try {
        const data: WarningEvent = JSON.parse(e.data);
        console.warn('[SSE] 后端警告:', data.code, data.message);
      } catch {
        // ignore parse errors
      }
    });

    eventSource.addEventListener('done', (e: MessageEvent) => {
      if (!mountedRef.current) return;
      clearAllTimers();
      try {
        const data: DoneEvent = JSON.parse(e.data);
        store.setDurationMs(data.duration_ms ?? 0);

        if (data.status === 'failed') {
          store.setJobStatus('failed');
          store.setSSEStatus('error');
          console.error('[SSE] 任务执行失败:', data.error_message);
        } else if (data.status === 'cancelled') {
          store.setJobStatus('cancelled');
          store.setSSEStatus('disconnected');
        } else {
          store.setJobStatus('completed');
          store.setSSEStatus('disconnected');
        }
      } catch (err) {
        console.warn('[SSE] 解析 done 事件失败:', err);
        store.setSSEStatus('disconnected');
      }
      eventSource.close();
    });

    eventSource.addEventListener('error', (e: Event) => {
      if (!mountedRef.current) return;
      // EventSource 原生 error 事件的 data 可能为空
      const msgEvent = e as MessageEvent;
      if (msgEvent.data) {
        try {
          const data: ErrorEvent = JSON.parse(msgEvent.data);
          console.error('[SSE] 服务端错误:', data.code, data.message);
        } catch {
          console.error('[SSE] 未知错误');
        }
      }
    });

    // EventSource 原生错误处理（网络断开等）
    eventSource.onerror = () => {
      if (!mountedRef.current) return;

      if (connectTimerRef.current) {
        clearTimeout(connectTimerRef.current);
        connectTimerRef.current = null;
      }

      if (eventSource.readyState === EventSource.CLOSED) {
        // 连接已关闭 → 尝试重连
        if (reconnectCountRef.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectCountRef.current += 1;
          const delay = RECONNECT_BASE_DELAY_MS * Math.pow(2, reconnectCountRef.current - 1);
          console.warn(
            `[SSE] 连接断开，${delay / 1000}s 后尝试第 ${reconnectCountRef.current}/${MAX_RECONNECT_ATTEMPTS} 次重连...`,
          );
          store.setSSEStatus('connecting');

          const timer = setTimeout(() => {
            if (mountedRef.current) {
              connectSSE();
            }
          }, delay);
          timerRef.current.push(timer);
        } else {
          console.error('[SSE] 已达最大重连次数，放弃重连');
          store.setSSEStatus('error');
        }
      }
      // EventSource.CONNECTING (0) / OPEN (1) / CLOSED (2)
    };

    return eventSource;
  }, [jobId, store, clearAllTimers, resetIdleTimer]);

  useEffect(() => {
    mountedRef.current = true;

    if (!jobId) return;

    store.reset();
    store.setSSEStatus('connecting');
    store.setStartedAt(Date.now());

    // 真实 SSE 连接
    connectSSE();

    return () => {
      mountedRef.current = false;
      clearAllTimers();
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [jobId]); // connectSSE 变更时不重建连接

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      clearAllTimers();
      timerRef.current.forEach(clearTimeout);
    };
  }, [clearAllTimers]);

  return store;
}

function dispatchSSEEvent(eventType: string, data: Record<string, unknown>) {
  const store = useReviewStore.getState();

  switch (eventType) {
    case 'progress':
      store.setProgress(data as unknown as ProgressEvent);
      break;
    case 'finding':
      store.addFinding(data as unknown as Finding);
      break;
    case 'chunk': {
      const chunk = data as unknown as ChunkEvent;
      if (chunk.target === 'summary') {
        store.addSummaryChunk(chunk.content);
      } else if (chunk.target === 'files') {
        try {
          const files = JSON.parse(chunk.content);
          if (Array.isArray(files)) store.setFileList(files);
        } catch {
          store.setFileList([chunk.content]);
        }
      } else {
        store.addReportChunk(chunk.content);
      }
      break;
    }
    case 'warning':
      break;
    case 'done': {
      const done = data as unknown as DoneEvent;
      store.setDurationMs(done.duration_ms ?? 0);
      store.setJobStatus(done.status === 'failed' ? 'failed' : 'completed');
      store.setSSEStatus('disconnected');
      break;
    }
    case 'error':
      console.error('[SSE] 错误:', (data as unknown as ErrorEvent).message);
      store.setSSEStatus('error');
      break;
    default:
      // 未识别的事件类型，尝试按 progress 处理
      if (data.step) {
        store.setProgress(data as unknown as ProgressEvent);
      }
  }
}
