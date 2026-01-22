"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/lib/auth-context";

const INITIAL_FORM = {
  email: "",
  password: "",
};

const ERROR_MESSAGES: Record<string, string> = {
  "auth/invalid-credential": "That email or password doesn’t match our records.",
  "auth/invalid-email": "Enter a valid email address.",
  "auth/user-not-found": "No account found for that email.",
  "auth/wrong-password": "That password isn’t correct.",
  "auth/email-already-in-use": "That email is already in use.",
  "auth/weak-password": "Use a stronger password with at least 6 characters.",
  "auth/popup-closed-by-user": "Google sign-in was canceled.",
  "auth/popup-blocked": "Allow popups to sign in with Google.",
};

type AuthMode = "signIn" | "signUp" | "reset";

export function AuthScreen() {
  const { signIn, signUp, resetPassword, signInWithGoogle } = useAuth();
  const [mode, setMode] = React.useState<AuthMode>("signIn");
  const [form, setForm] = React.useState(INITIAL_FORM);
  const [error, setError] = React.useState<string | null>(null);
  const [success, setSuccess] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);
  const isReset = mode === "reset";

  const formatError = (err: unknown) => {
    if (err && typeof err === "object" && "code" in err) {
      const code = String((err as { code?: string }).code ?? "");
      return ERROR_MESSAGES[code] ?? "We couldn’t sign you in. Try again.";
    }
    return err instanceof Error ? err.message : "We couldn’t sign you in. Try again.";
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setForm((prev) => ({ ...prev, [event.target.name]: event.target.value }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setSuccess(null);
    setSubmitting(true);

    try {
      if (mode === "signIn") {
        await signIn(form.email, form.password);
      }
      if (mode === "signUp") {
        await signUp(form.email, form.password);
      }
      if (mode === "reset") {
        await resetPassword(form.email);
        setSuccess("Password reset email sent.");
      }
      setForm(INITIAL_FORM);
    } catch (err) {
      setError(formatError(err));
    } finally {
      setSubmitting(false);
    }
  };

  const handleGoogle = async () => {
    setError(null);
    setSubmitting(true);
    try {
      await signInWithGoogle();
    } catch (err) {
      setError(formatError(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-dvh items-center justify-center bg-[radial-gradient(circle_at_top,_#f7f0ff,_#ffffff_55%,_#e8f6ff)] px-6 py-10 text-slate-900">
      <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white/80 p-8 shadow-lg backdrop-blur">
        <div className="mb-8 text-center">
          <p className="text-xs uppercase tracking-[0.4em] text-slate-500">
            LitLens
          </p>
          <h1 className="mt-3 text-2xl font-semibold text-slate-900">
            {mode === "signIn" && "Welcome back"}
            {mode === "signUp" && "Create your account"}
            {mode === "reset" && "Reset your password"}
          </h1>
          <p className="mt-2 text-sm text-slate-500">
            {mode === "reset"
              ? "We'll email you a reset link."
              : "Sign in to keep your research flowing."}
          </p>
        </div>

        <form className="space-y-4" onSubmit={handleSubmit}>
          {mode !== "reset" && (
            <button
              className="flex w-full items-center justify-center gap-2 rounded-md border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-900 shadow-sm transition hover:border-slate-300"
              type="button"
              onClick={handleGoogle}
              disabled={submitting}
            >
              <svg
                className="size-4"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  d="M23.49 12.27c0-.82-.07-1.41-.22-2.03H12v3.72h6.62c-.13.94-.85 2.36-2.44 3.32l-.02.12 3.56 2.76.25.02c2.3-2.12 3.62-5.23 3.62-7.91z"
                  fill="#4285F4"
                />
                <path
                  d="M12 24c3.24 0 5.95-1.06 7.94-2.88l-3.79-2.9c-1.01.71-2.38 1.2-4.15 1.2-3.18 0-5.88-2.12-6.84-5.05l-.11.01-3.68 2.85-.04.11C3.33 21.22 7.35 24 12 24z"
                  fill="#34A853"
                />
                <path
                  d="M5.16 14.37c-.25-.73-.39-1.5-.39-2.3s.14-1.57.38-2.3l-.01-.15-3.72-2.9-.12.06A12 12 0 0 0 0 12.07c0 1.94.47 3.77 1.32 5.39l3.84-3.09z"
                  fill="#FBBC05"
                />
                <path
                  d="M12 4.63c2.14 0 3.58.92 4.4 1.69l3.22-3.14C17.94 1.62 15.23 0 12 0 7.35 0 3.33 2.78 1.32 6.68l3.84 3.09c.96-2.93 3.66-5.14 6.84-5.14z"
                  fill="#EA4335"
                />
              </svg>
              <span>Login with Google</span>
            </button>
          )}
          <Input
            name="email"
            type="email"
            placeholder="Email"
            autoComplete="email"
            value={form.email}
            onChange={handleChange}
            required
          />
          {!isReset && (
            <Input
              name="password"
              type="password"
              placeholder="Password"
              autoComplete={mode === "signUp" ? "new-password" : "current-password"}
              value={form.password}
              onChange={handleChange}
              required
            />
          )}
          {error && (
            <p className="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-600">
              {error}
            </p>
          )}
          {success && (
            <p className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
              {success}
            </p>
          )}
          <Button className="w-full" type="submit" disabled={submitting}>
            {submitting
              ? "Working..."
              : mode === "signIn"
                ? "Sign in"
                : mode === "signUp"
                  ? "Create account"
                  : "Send reset link"}
          </Button>
        </form>

        <div className="mt-6 space-y-2 text-center text-sm text-slate-500">
          {mode !== "signIn" && (
            <button
              className="w-full rounded-md border border-slate-200 px-3 py-2 text-slate-700 shadow-sm transition hover:border-slate-300 hover:text-slate-900"
              onClick={() => setMode("signIn")}
              type="button"
            >
              Already have an account? Sign in
            </button>
          )}
          {mode !== "signUp" && (
            <button
              className="w-full rounded-md border border-slate-200 px-3 py-2 text-slate-700 shadow-sm transition hover:border-slate-300 hover:text-slate-900"
              onClick={() => setMode("signUp")}
              type="button"
            >
              New here? Create an account
            </button>
          )}
          {mode !== "reset" && (
            <button
              className="w-full rounded-md border border-slate-200 px-3 py-2 text-slate-700 shadow-sm transition hover:border-slate-300 hover:text-slate-900"
              onClick={() => setMode("reset")}
              type="button"
            >
              Forgot your password?
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
