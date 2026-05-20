export default {
  logo: <span style={{ fontWeight: 600 }}>TerminalWorld</span>,
  project: {
    link: 'https://github.com/AlienKevin/marin/tree/main/terminalworld-site',
  },
  docsRepositoryBase:
    'https://github.com/AlienKevin/marin/tree/main/terminalworld-site',
  footer: {
    content: (
      <span>
        TerminalWorld — neural terminal for agent training ·{' '}
        <a
          href="https://github.com/marin-community/marin/issues/5866"
          target="_blank"
          rel="noreferrer"
        >
          Issue #5866
        </a>
      </span>
    ),
  },
  head: (
    <>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <meta
        name="description"
        content="TerminalWorld — a neural world model for terminal agent training. Reports, experiments, and updates."
      />
      <meta property="og:title" content="TerminalWorld" />
      <meta
        property="og:description"
        content="A neural world model for terminal agent training."
      />
    </>
  ),
}
