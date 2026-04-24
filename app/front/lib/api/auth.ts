import { apiPost, apiPostForm, setAccessToken } from "./client";
import type { TokenResponse } from "./types";

export async function login(
  email: string,
  password: string,
): Promise<TokenResponse> {
  const form = new URLSearchParams({ username: email, password });
  const res = await apiPostForm<TokenResponse>("/api/auth/login", form);
  setAccessToken(res.access_token);
  return res;
}

export async function logout(): Promise<void> {
  await apiPost("/api/auth/logout");
  setAccessToken(null);
}
