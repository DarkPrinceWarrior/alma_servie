"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { status, user, logout } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (status === "anon") router.replace("/login");
  }, [status, router]);

  if (status !== "authed") {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-muted-foreground">
        {status === "loading" ? "Загрузка..." : "Перенаправление..."}
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-[#f9f9f9]">
      <header className="flex items-center justify-between border-b border-[#e5e5e5] bg-[#f9f9f9] pl-4">
        <div className="flex items-center gap-2 pr-4 text-sm text-[#424247]">
          <span className="font-medium">Alma · Аномалии</span>
        </div>
        <div className="flex items-center gap-3 pr-4 text-sm">
          <span className="text-[#797979]">{user.email}</span>
          <button
            type="button"
            onClick={() => logout()}
            className="rounded-[8px] border border-[#e5e5e5] bg-white px-3 py-1 text-[#424247] hover:bg-[#f3f3f3]"
          >
            Выйти
          </button>
        </div>
      </header>
      <main className="flex-1">{children}</main>
    </div>
  );
}
