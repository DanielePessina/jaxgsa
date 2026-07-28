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
        { text: 'Getting Started', link: '/guide/getting-started' },
        { text: 'Migrating to 0.4', link: '/guide/migration-0.4' },
        { text: 'Methods', link: '/guide/methods' },
        { text: 'Benchmarks', link: '/guide/benchmarks' },
        { text: 'Configuration', link: '/guide/configuration' },
      ],
      '/examples/': [
        { text: 'Basic (Ishigami)', link: '/examples/basic' },
        { text: 'Non-Uniform Inputs', link: '/examples/non-uniform-inputs' },
        { text: 'Correlated Inputs', link: '/examples/correlated-inputs' },
        { text: 'Categorical Inputs', link: '/examples/categorical-inputs' },
        { text: 'Save & Reload', link: '/examples/save-load' },
        { text: 'Bootstrap CIs', link: '/examples/bootstrap' },
        { text: 'Multi-Output & Time-Series', link: '/examples/multi-output' },
        { text: 'xarray Output', link: '/examples/xarray' },
        { text: 'RS-HDMR', link: '/examples/hdmr' },
        { text: 'Advanced Workflow', link: '/examples/advanced-workflow' },
        { text: 'PCE', link: '/examples/pce' },
        { text: 'Shapley Effects', link: '/examples/shapley' },
        { text: 'eFAST', link: '/examples/efast' },
        { text: 'DGSM', link: '/examples/dgsm' },
        { text: 'Morris', link: '/examples/morris' },
        { text: 'HSIC', link: '/examples/hsic' },
        { text: 'PAWN', link: '/examples/pawn' },
        { text: 'Borgonovo Delta', link: '/examples/borgonovo' },
        { text: 'Optimal Transport', link: '/examples/optimal-transport' },
        { text: 'Batch Reactor (notebook)', link: '/examples/batch_reactor' },
      ],
      '/api/': [
        { text: 'API Reference', link: '/api/' },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/danielepessina/jaxgsa' },
    ],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under the MIT License.',
    },
  },
})
