import { apiGet } from "./client";
import type { UserRead } from "./types";

export async function getMe(): Promise<UserRead> {
  return apiGet<UserRead>("/api/users/me");
}
