import { useEffect, useState } from "react";

/**
 * Prefix of every localStorage key the studio autosaves. The session keeps
 * only small settings (problem rows, form fields, method selection); uploaded
 * data and results are never stored.
 */
const PREFIX = "jaxgsa-studio:";

/** Bump when a stored shape changes, so stale entries are ignored. */
const VERSION = 1;

interface Stored<T> {
  v: number;
  value: T;
}

function read<T>(key: string, initial: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key);

    if (raw === null) return initial;

    // SAFETY: only `write` stores under PREFIX, always as `Stored<T>`; the
    // version check below rejects entries from an older shape.
    const parsed = JSON.parse(raw) as Stored<T>;

    return parsed.v === VERSION ? parsed.value : initial;
  } catch {
    // Private browsing or corrupt JSON: start from the default.
    return initial;
  }
}

function write<T>(key: string, value: T): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify({ v: VERSION, value } satisfies Stored<T>));
  } catch {
    // Storage blocked or full: the page still works, it just forgets.
  }
}

/**
 * `useState` that autosaves to localStorage under `key`.
 *
 * The value must be JSON-serialisable. A missing, blocked or stale entry
 * falls back to `initial`.
 */
export function usePersistentState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => read(key, initial));

  useEffect(() => {
    write(key, value);
  }, [key, value]);

  return [value, setValue] as const;
}

/** Read a persisted value once, outside React (e.g. to seed a reducer). */
export function readPersisted<T>(key: string, initial: T): T {
  return read(key, initial);
}

/** Save a value outside React (e.g. from a reducer effect). */
export function writePersisted<T>(key: string, value: T): void {
  write(key, value);
}

/** Remove every autosaved studio key (the "reset session" button). */
export function clearPersistedSession(): void {
  try {
    const keys: string[] = [];

    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);

      if (k?.startsWith(PREFIX)) keys.push(k);
    }

    keys.forEach((k) => localStorage.removeItem(k));
  } catch {
    // Nothing to clear when storage is blocked.
  }
}
