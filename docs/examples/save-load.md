# Save and Reload Samples

Use `SamplingResult.save()` when you want to generate Sobol/Saltelli samples
once, store them on disk, and reuse them later without re-running the sampling
step.

## Save, reload, and analyze

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)

# Save a file stem, not a full filename with extension.
sampling_result.save("runs/ishigami", format="csv")

restored = gsax.load("runs/ishigami", format="csv")
Y = evaluate(restored.samples)
result = gsax.analyze(restored, Y)

print("Loaded rows:", restored.n_total)
print("First SampleID values:")
print(restored.samples_df.head())
print(result.S1)
print(result.ST)
```

## Files written to disk

For `path="runs/ishigami"` and `format="csv"`, `save()` writes:

- `runs/ishigami.csv` with the unique sample matrix.
- `runs/ishigami.json` with the `Problem` definition, `base_n`,
  `expanded_n_total`, and related metadata.
- `runs/ishigami.npz` only when `expanded_to_unique` is not the identity map.

The `.npz` file is skipped for storage efficiency when the expanded Saltelli
rows already map one-to-one to the unique rows.

## Choosing a format

- `csv` is the safest default for interop and version control.
- `txt` is useful for plain numeric pipelines.
- `pkl` is compact inside Python-only workflows.
- `xlsx` requires `openpyxl`.
- `parquet` requires `pyarrow`.

Use the same `format` when calling `gsax.load()`. Format is not inferred from
the metadata file.

## Practical caveats

- Saving persists the metadata that `gsax.analyze()` needs; do not drop the
  `.json` file.
- Reloaded samples are still the unique rows, so your model evaluation step
  remains `Y = model(restored.samples)`.
- If you need a human-readable audit trail, `samples_df` is easier to inspect
  than the raw NumPy array.

## See also

- [Basic Example](/examples/basic) for the minimal Sobol workflow.
- [Bootstrap Confidence Intervals](/examples/bootstrap) if you want uncertainty
  bounds on a reloaded design.
- [Advanced Workflow](/examples/advanced-workflow) for a larger custom model
  that keeps Sobol and HDMR runs side by side.
