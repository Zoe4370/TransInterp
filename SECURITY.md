# Security Policy

TransInterp is maintained by **Zoe Faith Gumise**. Please use the private contact options below for vulnerability reports rather than opening a public issue.

## Supported versions

TransInterp is pre-1.0. Security fixes are made against the latest release
on the default branch; there is no long-term support branch yet.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Instead,
use GitHub's private vulnerability reporting (Security tab → "Report a
vulnerability") on this repository, or email gumisezoe@gmail.com.
Include a minimal reproduction and the affected version.

We aim to acknowledge reports within five business days and to agree on a
disclosure timeline with the reporter before any public write-up.

## Scope notes

TransInterp loads model weights and configuration files supplied by the
user. As with any tool that deserializes model artifacts, only load weights
and YAML/JSON configuration from sources you trust; `trust_remote_code` in
`ModelConfig` defaults to `false` for this reason.
