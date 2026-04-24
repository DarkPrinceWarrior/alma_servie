"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { auth, type UserRead, users } from "@/lib/api";
import { getAccessToken, setAccessToken } from "@/lib/api/client";

export type AuthState =
  | { status: "loading"; user: null }
  | { status: "anon"; user: null }
  | { status: "authed"; user: UserRead };

export function useAuth() {
  const [state, setState] = useState<AuthState>({
    status: "loading",
    user: null,
  });
  const router = useRouter();

  useEffect(() => {
    let active = true;
    const token = getAccessToken();
    if (!token) {
      setState({ status: "anon", user: null });
      return;
    }
    users
      .getMe()
      .then((u) => {
        if (active) setState({ status: "authed", user: u });
      })
      .catch(() => {
        if (active) {
          setAccessToken(null);
          setState({ status: "anon", user: null });
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      await auth.login(email, password);
      const u = await users.getMe();
      setState({ status: "authed", user: u });
      router.push("/");
    },
    [router],
  );

  const logout = useCallback(async () => {
    try {
      await auth.logout();
    } finally {
      setAccessToken(null);
      setState({ status: "anon", user: null });
      router.push("/login");
    }
  }, [router]);

  return { ...state, login, logout };
}
