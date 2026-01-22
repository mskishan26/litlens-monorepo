"use client";

import { Assistant } from "@/app/assistant";
import { AuthScreen } from "@/components/auth/auth-screen";
import { useAuth } from "@/lib/auth-context";

export function AuthenticatedApp() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-slate-950 text-white">
        <div className="text-sm uppercase tracking-[0.4em]">Loading</div>
      </div>
    );
  }

  if (!user) {
    return <AuthScreen />;
  }

  return <Assistant />;
}
