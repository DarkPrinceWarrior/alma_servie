// fetch wrapper with access token + refresh rotation on 401.
// Refresh token lives in httpOnly cookie scoped to /api/auth/refresh —
// Next.js rewrites proxy same-origin so the cookie is sent automatically.

let accessToken: string | null = null;
let refreshing: Promise<boolean> | null = null;

export function setAccessToken(token: string | null) {
  accessToken = token;
  if (typeof window !== "undefined") {
    if (token) {
      sessionStorage.setItem("alma_at", token);
    } else {
      sessionStorage.removeItem("alma_at");
    }
  }
}

export function getAccessToken(): string | null {
  if (accessToken) return accessToken;
  if (typeof window !== "undefined") {
    accessToken = sessionStorage.getItem("alma_at");
  }
  return accessToken;
}

async function tryRefresh(): Promise<boolean> {
  if (refreshing) return refreshing;
  refreshing = (async () => {
    const res = await fetch("/api/auth/refresh", {
      method: "POST",
      credentials: "include",
    });
    if (!res.ok) {
      setAccessToken(null);
      return false;
    }
    const body = (await res.json()) as { access_token: string };
    setAccessToken(body.access_token);
    return true;
  })();
  try {
    return await refreshing;
  } finally {
    refreshing = null;
  }
}

export interface ApiError extends Error {
  status: number;
  body: unknown;
}

function makeError(status: number, body: unknown): ApiError {
  const err = new Error(
    typeof body === "object" && body && "detail" in body
      ? String((body as { detail: unknown }).detail)
      : `HTTP ${status}`,
  ) as ApiError;
  err.status = status;
  err.body = body;
  return err;
}

async function request(
  path: string,
  init: RequestInit = {},
  { retried = false }: { retried?: boolean } = {},
): Promise<Response> {
  const headers = new Headers(init.headers ?? {});
  const token = getAccessToken();
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const res = await fetch(path, {
    ...init,
    headers,
    credentials: "include",
  });

  if (res.status === 401 && !retried && !path.includes("/api/auth/")) {
    const ok = await tryRefresh();
    if (ok) return request(path, init, { retried: true });
  }

  return res;
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await request(path, { method: "GET" });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw makeError(res.status, body);
  return body as T;
}

export async function apiPostJson<T>(
  path: string,
  payload: unknown,
): Promise<T> {
  const res = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw makeError(res.status, body);
  return body as T;
}

export async function apiPostForm<T>(
  path: string,
  form: URLSearchParams,
): Promise<T> {
  const res = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form.toString(),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw makeError(res.status, body);
  return body as T;
}

export async function apiPostMultipart<T>(
  path: string,
  form: FormData,
): Promise<T> {
  // No explicit Content-Type — the browser sets the multipart boundary.
  const res = await request(path, { method: "POST", body: form });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw makeError(res.status, body);
  return body as T;
}

export async function apiPost(path: string): Promise<void> {
  const res = await request(path, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw makeError(res.status, body);
  }
}

export { request };
