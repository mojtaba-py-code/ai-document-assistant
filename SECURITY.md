# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through
GitHub's [private vulnerability reporting](https://github.com/mojtaba-py-code/ai-document-assistant/security/advisories/new).

Include the affected version or commit, a description of the impact, and steps to reproduce.
You will receive an acknowledgement within 3 working days and an assessment within 10.
Coordinated disclosure: fixes are released before details are published, and reporters are
credited unless they prefer otherwise.

## Scope

In scope: the `docassist` package, its migrations, the web UI, the container image and the
deployment manifests in this repository. Especially interesting:

* cross-tenant or cross-department data access (including via search, embeddings, the AI
  assistant, exports or audit records);
* authentication or session bypasses, privilege escalation;
* prompt-injection techniques that cause an action, a data leak or a fabricated citation
  despite the controls described in [docs/security-architecture.md](docs/security-architecture.md);
* malicious-file handling (sandbox escape, scanner bypass, zip/XML bombs);
* SSRF, SQL injection, XSS, CSRF.

Out of scope: findings that require a compromised host or database superuser, missing
hardening in your own infrastructure, and the documented residual risks in
[docs/threat-model.md](docs/threat-model.md#9-assumptions-and-residual-risks).

## Supported versions

Only the latest release receives security fixes.

## Security documentation

* [Threat model](docs/threat-model.md)
* [Security architecture](docs/security-architecture.md)
* [Operations runbook (incident response, key rotation)](docs/operations.md)
* [Final security audit](docs/security-audit.md)
