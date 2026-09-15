import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'jaxgsa',
  description: 'Global Sensitivity Analysis in JAX',
  base: '/jaxgsa/',
  markdown: {
    math: true,
  },

  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Examples', link: '/examples/basic' },
      { text: 'API', link: '/api/' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Start here',
          items: [
            { text: 'Getting started', link: '/guide/getting-started' },
            { text: 'Choosing a method', link: '/guide/methods' },
          ],
        },
        {
          text: 'Understand',
          items: [
            { text: 'Concepts', link: '/guide/concepts' },
            { text: 'Dependent inputs', link: '/guide/dependent-inputs' },
            {
              text: 'Differentiating analyses',
              link: '/guide/differentiation',
            },
          ],
        },
        {
          text: 'Plan and configure',
          items: [
            { text: 'Scaling to large problems', link: '/guide/scale' },
            { text: 'Benchmarks', link: '/guide/benchmarks' },
            { text: 'Configuration', link: '/guide/configuration' },
          ],
        },
      ],
      '/examples/': [
        {
          text: 'Start here',
          items: [{ text: 'Basic (Ishigami)', link: '/examples/basic' }],
        },
        {
          text: 'Inputs and outputs',
          items: [
            { text: 'Non-uniform inputs', link: '/examples/non-uniform-inputs' },
            { text: 'Correlated inputs', link: '/examples/correlated-inputs' },
            { text: 'Categorical inputs', link: '/examples/categorical-inputs' },
            { text: 'Multi-output and time series', link: '/examples/multi-output' },
            { text: 'xarray output', link: '/examples/xarray' },
          ],
        },
        {
          text: 'Workflows',
          items: [
            { text: 'Screen first, then quantify', link: '/examples/advanced-workflow' },
            { text: 'Method comparison', link: '/examples/method-comparison' },
            { text: 'Bootstrap intervals', link: '/examples/bootstrap' },
            { text: 'Save and reload', link: '/examples/save-load' },
          ],
        },
        {
          text: 'Methods',
          items: [
            { text: 'Morris', link: '/examples/morris' },
            { text: 'eFAST', link: '/examples/efast' },
            { text: 'DGSM', link: '/examples/dgsm' },
            { text: 'Kucherenko', link: '/examples/kucherenko' },
            { text: 'PCE', link: '/examples/pce' },
            { text: 'RS-HDMR', link: '/examples/hdmr' },
            { text: 'Shapley effects', link: '/examples/shapley' },
            { text: 'VKOGA', link: '/examples/vkoga' },
            { text: 'HSIC', link: '/examples/hsic' },
            { text: 'PAWN', link: '/examples/pawn' },
            { text: 'Borgonovo delta', link: '/examples/borgonovo' },
            { text: 'Optimal transport', link: '/examples/optimal-transport' },
          ],
        },
        {
          text: 'Applications',
          items: [
            { text: 'Batch reactor', link: '/examples/batch_reactor' },
          ],
        },
      ],
      // Every method page used to be reachable only through links inside
      // /api/index.md, so the sidebar hid two thirds of the reference.
      '/api/': [
        { text: 'Overview', link: '/api/' },
        {
          text: 'Setting up a study',
          items: [
            { text: 'Problem', link: '/api/problem' },
            { text: 'Sampling', link: '/api/sampling' },
          ],
        },
        {
          text: 'Methods that build a design',
          items: [
            { text: 'Sobol', link: '/api/sobol' },
            { text: 'Morris', link: '/api/morris' },
            { text: 'eFAST', link: '/api/efast' },
            { text: 'DGSM', link: '/api/dgsm' },
            { text: 'Kucherenko', link: '/api/kucherenko' },
          ],
        },
        {
          text: 'Methods for given data',
          items: [
            { text: 'HSIC', link: '/api/hsic' },
            { text: 'PAWN', link: '/api/pawn' },
            { text: 'Borgonovo delta', link: '/api/borgonovo' },
            { text: 'Optimal transport', link: '/api/optimal-transport' },
          ],
        },
        {
          text: 'Surrogate methods',
          items: [
            { text: 'PCE', link: '/api/pce' },
            { text: 'RS-HDMR', link: '/api/hdmr' },
            { text: 'Shapley effects', link: '/api/shapley' },
            { text: 'VKOGA', link: '/api/vkoga' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/danielepessina/jaxgsa' },
    ],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under the BSD-3-Clause License.',
    },
  },
})
