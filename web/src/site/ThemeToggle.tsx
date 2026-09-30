import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

const STORAGE_KEY = "jaxgsa-theme";

const DARK_QUERY = "(prefers-color-scheme: dark)";

/** Switches between light and dark themes, following the device until chosen. */
export function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));

  useEffect(() => {
    const preference = window.matchMedia(DARK_QUERY);

    const onSystemChange = (event: MediaQueryListEvent) => {
      try {
        if (localStorage.getItem(STORAGE_KEY) !== null) return;
      } catch {
        // Private browsing can block storage; still follow the device theme.
      }

      document.documentElement.classList.toggle("dark", event.matches);
      setDark(event.matches);
    };

    preference.addEventListener("change", onSystemChange);

    return () => preference.removeEventListener("change", onSystemChange);
  }, []);

  const toggle = () => {
    const nextDark = !dark;
    document.documentElement.classList.toggle("dark", nextDark);
    setDark(nextDark);

    try {
      localStorage.setItem(STORAGE_KEY, nextDark ? "dark" : "light");
    } catch {
      // The choice still works for this page when storage is unavailable.
    }
  };

  return (
    <Button
      type="button"
      variant="outline"
      size="icon"
      onClick={toggle}
      aria-label={`Switch to ${dark ? "light" : "dark"} mode`}
      title={`Switch to ${dark ? "light" : "dark"} mode`}
    >
      {dark ? <Sun aria-hidden="true" /> : <Moon aria-hidden="true" />}
    </Button>
  );
}
