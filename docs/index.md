---
layout: home

hero:
  name: jaxgsa
  text: Find which inputs matter
  tagline: Global sensitivity analysis for computational models. Explore how changing your inputs changes the output, then choose a method that fits your question and data.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: Choose a Method
      link: /guide/methods
    - theme: alt
      text: API Reference
      link: /api/problem
features:
  - title: Start with one result
    details: Define the inputs, run the model, and read which inputs matter on their own or through interactions.
  - title: Choose a useful measure
    details: Screen many inputs, measure shares of output variation, or study how inputs change the output distribution.
  - title: Use the data you have
    details: Some methods choose new model runs. Others analyze input and output samples you already collected.
  - title: Go beyond one output
    details: Analyze several outputs or a time series, including channels measured on different time grids.
---

## Your first analysis

Suppose a model has three uncertain inputs. You want to know which ones explain
changes in its output. The [getting started guide](/guide/getting-started) takes
you from defining those inputs to interpreting a Sobol result. In that result,
`S1` measures an input's contribution on its own; `ST` also includes its
interactions with other inputs. A large gap between them tells you that the
input matters in combination with others.

After the first run, [Concepts](/guide/concepts) explains the ideas behind
those numbers. The [basic example](/examples/basic) shows how to check whether
the estimates have settled as you add samples.

## Where to go next

- [Choose a method](/guide/methods) when you know the question but are unsure
  which analysis to run.
- [Use non-uniform inputs](/examples/non-uniform-inputs) when a range alone does
  not describe how likely each input value is.
- [Plan a larger study](/guide/scale) when model evaluations are expensive or
  you have many inputs.
- [Look up the API](/api/) for arguments, return values, and supported input
  types.

## Performance

Performance depends on the method, sample count, number of inputs, and output
shape. See the [benchmarks guide](/guide/benchmarks) for measured workloads and
hardware.

jaxgsa's Sobol sampling and analysis workflow follows
[SALib](https://salib.readthedocs.io/), reimplemented for JAX.
