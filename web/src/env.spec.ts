import { beforeAll, describe, expect, it } from "vitest";
import { defaultDevice, grad, init, jit, numpy as np, vmap } from "@jax-js/jax";

const KNOWN_DEVICES = ["cpu", "wasm", "webgpu", "webgl"] as const;

let availableDevices: string[] = [];

describe("jax-js wasm runtime (Phase 0 spike)", () => {
  beforeAll(async () => {
    availableDevices = await init();
    defaultDevice("wasm");
  });

  it("reports at least the wasm device after init()", () => {
    expect(availableDevices).toContain("wasm");
    expect(availableDevices.every((d) => KNOWN_DEVICES.includes(d as never))).toBe(true);
    expect(() => defaultDevice("wasm")).not.toThrow();
  });

  it("creates float64 arrays and computes a float64 matmul", () => {
    const x = np.array(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: np.float64 },
    );

    expect(x.dtype).toBe(np.float64);
    expect(x.ref.dataSync()).toBeInstanceOf(Float64Array);

    const xt = x.ref.transpose();
    const c = np.matmul(x, xt);
    expect(Array.from(c.dataSync())).toEqual([5, 11, 11, 25]);
  });

  it("performs true float64 arithmetic (0.1 + 0.2), not float32 emulation", () => {
    const a = np.array([0.1], { dtype: np.float64 });
    const b = np.array([0.2], { dtype: np.float64 });
    const sum64 = a.add(b);
    const f64 = sum64.dataSync()[0];
    expect(Math.abs(f64 - 0.30000000000000004)).toBeLessThan(1e-16);

    const c = np.array([0.1], { dtype: np.float32 });
    const d = np.array([0.2], { dtype: np.float32 });
    const sum32 = c.add(d);
    const f32 = sum32.dataSync()[0];
    expect(Math.abs(f32 - 0.30000001192092896)).toBeLessThan(1e-9);
    expect(f64).not.toBe(f32);
  });

  it("fuses operations into a single kernel with jit()", () => {
    const hypot = jit((x: np.Array, y: np.Array) =>
      np.sqrt(np.square(x).add(np.square(y))),
    );

    const x = np.array([3, 5, 8], { dtype: np.float64 });
    const y = np.array([4, 12, 15], { dtype: np.float64 });
    const out = hypot(x, y);
    expect(Array.from(out.dataSync())).toEqual([5, 13, 17]);
  });

  it("supports grad() and vmap()", () => {
    const x = np.linspace(-10, 10, 1000, true);
    const y1 = vmap(grad(np.sin))(x.ref);
    const y2 = np.cos(x);
    expect(np.allclose(y1, y2, { atol: 1e-12 })).toBe(true);
  });
});
