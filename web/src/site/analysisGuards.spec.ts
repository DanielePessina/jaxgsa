import { describe, expect, it } from "vitest";
import {
  EFAST_LOW_N,
  createRunGuard,
  efastLowNWarning,
  parseOrder,
  pceFitWarnings,
  type SliceFit,
} from "./analysisGuards";

describe("parseOrder", () => {
  it("accepts whole numbers of 1 or more", () => {
    expect(parseOrder("1")).toEqual({ ok: true, value: 1 });
    expect(parseOrder(" 8 ")).toEqual({ ok: true, value: 8 });
  });

  it.each(["", "   ", "0", "-2", "2.5", "1e1", "abc", "NaN", "9999999999999999999"])(
    "rejects %j",
    (text) => {
      const parsed = parseOrder(text);

      expect(parsed.ok).toBe(false);

      if (!parsed.ok) expect(parsed.message).toMatch(/order/i);
    },
  );
});

describe("createRunGuard", () => {
  it("keeps the newest token current", () => {
    const guard = createRunGuard();
    const token = guard.start();

    expect(guard.isCurrent(token)).toBe(true);
  });

  it("makes a token stale after cancel", () => {
    const guard = createRunGuard();
    const token = guard.start();
    guard.cancel();

    expect(guard.isCurrent(token)).toBe(false);
  });

  it("makes an older run stale when a new run starts", () => {
    const guard = createRunGuard();
    const first = guard.start();
    const second = guard.start();

    expect(guard.isCurrent(first)).toBe(false);
    expect(guard.isCurrent(second)).toBe(true);
  });

  it("drops results from a loop whose data changed mid-run", async () => {
    const guard = createRunGuard();
    const stored: string[] = [];

    const run = async (methods: string[], onWait: (i: number) => void) => {
      const token = guard.start();

      for (let i = 0; i < methods.length; i++) {
        await Promise.resolve();
        onWait(i);

        if (!guard.isCurrent(token)) return;
        stored.push(methods[i]);
      }
    };

    // The user changes Y while the loop waits before its second method.
    await run(["pce", "pawn", "borgonovo"], (i) => {
      if (i === 1) guard.cancel();
    });

    expect(stored).toEqual(["pce"]);
  });
});

const fit = (label: string, explainedVariance: number, cvRmse: number, stdY = 1): SliceFit => ({
  label,
  explainedVariance,
  cvRmse,
  stdY,
});

const same = { requested: 3, effective: 3, nRows: 200 };

describe("pceFitWarnings", () => {
  it("is silent for a good fit", () => {
    expect(pceFitWarnings([fit("y", 0.99, 0.1)], same)).toEqual([]);
  });

  it("warns below the 0.5 explained variance line, naming the worst slice", () => {
    const out = pceFitWarnings([fit("a", 0.9, 0.2), fit("b", 0.25, 0.9)], same);

    expect(out).toHaveLength(1);
    expect(out[0]).toMatch(/Explained variance is 0\.25 on b, below 0\.5/);
    expect(out[0]).toMatch(/more than half of the output variance unexplained/);
  });

  it("does not warn at exactly 0.5", () => {
    expect(pceFitWarnings([fit("y", 0.5, 0.1)], same)).toEqual([]);
  });

  it("warns about an overfit when LOO passes sqrt(0.5) of std(Y)", () => {
    // Python case: in-sample 0.78 fine, LOO 0.78 x std(Y).
    const out = pceFitWarnings([fit("y", 0.506873, 2.879327, 3.670963)], same);

    expect(out).toHaveLength(1);
    expect(out[0]).toMatch(/leave-one-out error is 0\.78 times/);
  });

  it("gives one warning, not two, when the in-sample fit already failed", () => {
    const out = pceFitWarnings([fit("y", 0.25, 0.92)], same);

    expect(out).toHaveLength(1);
    expect(out[0]).toMatch(/Explained variance/);
  });

  it("skips slices that are not finite", () => {
    const out = pceFitWarnings([fit("y", NaN, NaN, 0)], same);

    expect(out).toEqual([]);
  });

  it("reports a lowered order", () => {
    const out = pceFitWarnings([fit("y", 0.99, 0.1)], { requested: 10, effective: 8, nRows: 100 });

    expect(out).toHaveLength(1);
    expect(out[0]).toMatch(/lowered from 10 to 8/);
  });
});

describe("efastLowNWarning", () => {
  it("sets the threshold at 257", () => {
    expect(EFAST_LOW_N).toBe(257);
    expect(efastLowNWarning(256)).not.toBeNull();
  });

  it("warns for the default 129 points and cites the measured bias", () => {
    expect(efastLowNWarning(129)).toMatch(/0\.31 to 0\.81/);
  });

  it("is silent at and above the threshold", () => {
    expect(efastLowNWarning(EFAST_LOW_N)).toBeNull();
    expect(efastLowNWarning(1025)).toBeNull();
  });
});
