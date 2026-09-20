import { useCallback, useState } from "react";

/**
 * The busy/error dance around synchronous wasm compute: yield to paint
 * (the jax-js calls block the main thread), then run, catching errors into
 * a shared error state. Returns true on success, false on error.
 */
export function useBusyAction() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (fn: () => void): Promise<boolean> => {
    setError(null);
    setBusy(true);
    await new Promise((r) => setTimeout(r, 10));

    try {
      fn();

      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));

      return false;
    } finally {
      setBusy(false);
    }
  }, []);

  return { busy, error, setError, run };
}