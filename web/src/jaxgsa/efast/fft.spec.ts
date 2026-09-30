import { describe, expect, it } from "vitest";
import { defaultDevice, init, numpy as np } from "@jax-js/jax";

describe("jax-js fft smoke test", () => {
  it("computes rfft magnitudes on the wasm device", async () => {
    await init();
    defaultDevice("wasm");

    // A pure cosine: real spectrum with a peak at bin k=3.
    const N = 256;
    const y = new Float64Array(N);

    for (let i = 0; i < N; i++) y[i] = Math.cos((2 * Math.PI * 3 * i) / N);

    // SAFETY: y is a freshly allocated host Float64Array with N entries.
    const spec = np.fft.rfft(np.array(y as Float64Array<ArrayBuffer>, { dtype: np.float64 }), 0);

    const mag = np.hypot(spec.real, spec.imag);

    // SAFETY: jax-js dataSync returns a host Float64Array for the float64 magnitude.
    const out = mag.dataSync() as Float64Array;

    expect(out.length).toBe(N / 2 + 1);

    // Peak at bin 3 (amplitude N/2), near zero elsewhere.
    expect(out[3]).toBeCloseTo(N / 2, 4);

    const maxOther = Math.max(...Array.from(out).filter((_, i) => i !== 3 && i !== 0));
    expect(maxOther).toBeLessThan(1e-3);
  });
});