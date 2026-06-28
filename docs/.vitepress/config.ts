import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'gsax',
  description: 'Global Sensitivity Analysis in JAX',
  base: '/gsax/',
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
        { text: 'Methods', link: '/guide/methods' },
        { text: 'Benchmarks', link: '/guide/benchmarks' },
      ],
      '/examples/': [
        { text: 'Basic (Ishigami)', link: '/examples/basic' },
        { text: 'Non-Uniform Inputs', link: '/examples/non-uniform-inputs' },
        { text: 'Save & Reload', link: '/examples/save-load' },
        { text: 'Bootstrap CIs', link: '/examples/bootstrap' },
        { text: 'Multi-Output & Time-Series', link: '/examples/multi-output' },
        { text: 'xarray Output', link: '/examples/xarray' },
        { text: 'RS-HDMR', link: '/examples/hdmr' },
        { text: 'Advanced Workflow', link: '/examples/advanced-workflow' },
        { text: 'eFAST', link: '/examples/efast' },
        { text: 'DGSM', link: '/examples/dgsm' },
        { text: 'PCE', link: '/examples/pce' },
        { text: 'HSIC', link: '/examples/hsic' },
        { text: 'PAWN', link: '/examples/pawn' },
        { text: 'Batch Reactor (notebook)', link: '/examples/batch_reactor' },
      ],
      '/api/': [
        { text: 'API Reference', link: '/api/' },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/danielepessina/gsax' },
    ],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under the MIT License.',
    },
  },
})
