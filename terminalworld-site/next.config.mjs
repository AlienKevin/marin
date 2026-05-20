import nextra from 'nextra'

const withNextra = nextra({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.jsx',
})

export default withNextra({
  async redirects() {
    return [
      { source: '/literature-review', destination: '/', permanent: false },
      { source: '/world-model-evaluation', destination: '/', permanent: false },
    ]
  },
})
