# Security and Integrity Policy

Version `3.x` receives integrity and security fixes. Do not report general model limitations as software security vulnerabilities. Report exploitable issues privately through a GitHub Security Advisory for the repository, or through the maintainer's approved private security channel when the repository is hosted elsewhere. Do not put positions, credentials, proprietary data, or exploit details in a public issue.

Operational deployments must provide secrets through an approved secret manager; this repository requires no credentials for its public H.15/FRED source. Never add API keys, positions, client information, or internal limits to source control.

The package treats data and output integrity as security properties: immutable snapshot hashes, bounded network responses, crash-safe platform locks, atomic cache/output writes, content-addressed identities, artifact schema contracts, and hash readback. The local JSONL execution ledger is concurrency-controlled on POSIX but is not signed, WORM, or an enterprise system of record. Deployment owners remain responsible for authentication, authorization, network policy, dependency/SBOM scanning, signed releases, centralized immutable audit retention, backup/restore, and incident response.
