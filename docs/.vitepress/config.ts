import { defineConfig } from 'vitepress'
import type { HeadConfig } from 'vitepress'

const siteUrl = 'https://danielepessina.github.io/jaxgsa'

const description = 'Global Sensitivity Analysis in JAX'

const softwareSchema = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareSourceCode',
  name: 'jaxgsa',
  description,
  codeRepository: 'https://github.com/DanielePessina/jaxgsa',
  programmingLanguage: 'Python',
  runtimePlatform: 'Python 3.12+',
  license: 'https://github.com/DanielePessina/jaxgsa/blob/master/LICENSE',
  url: siteUrl,
  sameAs: [
    'https://pypi.org/project/jaxgsa/',
    'https://doi.org/10.5281/zenodo.22099117',
    'https://github.com/DanielePessina/jaxgsa',
  ],
  author: [
    { '@type': 'Person', name: 'Daniele Pessina' },
    { '@type': 'Person', name: 'Maria M. Papathanasiou' },
  ],
}

function pageUrl(relativePath: string): string {
  if (relativePath === 'index.md') return siteUrl
  return `${siteUrl}${relativePath.replace(/\.md$/, '.html')}`
}

export default defineConfig({
  title: 'jaxgsa',
  description,
  base: '/jaxgsa/',
  markdown: {
    math: true,
  },

  head: [
    ['meta', { name: 'google-site-verification', content: 'google05262a1df685974f' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'jaxgsa' }],
    ['meta', { name: 'twitter:card', content: 'summary' }],
  ],

  sitemap: {
    hostname: siteUrl,
    transformItems(items) {
      return [...items, { url: `${siteUrl}workbench/` }]
    },
  },

  transformHead({ pageData, title, description: pageDescription }) {
    const head: HeadConfig[] = [
      ['link', { rel: 'canonical', href: pageUrl(pageData.relativePath) }],
      ['meta', { property: 'og:title', content: title }],
      ['meta', { property: 'og:description', content: pageDescription || description }],
      ['meta', { property: 'og:url', content: pageUrl(pageData.relativePath) }],
    ]
    if (pageData.relativePath === 'index.md') {
      head.push(['script', { type: 'application/ld+json' }, JSON.stringify(softwareSchema)])
    }
    return head
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
            { text: 'Irregular output grids', link: '/examples/irregular-outputs' },
            { text: 'xarray output', link: '/examples/xarray' },
          ],
        },
        {
          text: 'Workflows',
          items: [
            { text: 'Screen first, then quantify', link: '/examples/advanced-workflow' },
            { text: 'Method comparison', link: '/examples/method-comparison' },
            { text: 'Bootstrap intervals', link: '/examples/bootstrap' },
            { text: 'Fit input ranges', link: '/examples/inverse-sobol-bounds' },
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
            { text: 'Kucherenko', link: '/api/kucherenko' },
          ],
        },
        {
          text: 'Gradient methods',
          items: [
            { text: 'DGSM', link: '/api/dgsm' },
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
