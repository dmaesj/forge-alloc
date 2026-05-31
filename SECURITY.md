# Security Policy

`forge-alloc` is a memory-allocator library whose explicit purpose includes
security hardening (guard pages, freelist MACs, canaries, poison-on-free,
quarantine). A defect in this code can directly weaken the memory safety of
everything built on top of it, so vulnerability reports are taken seriously.

## Supported versions

Security fixes are issued for the latest published `0.x` release line of each
crate. Because the project is pre-1.0, fixes are delivered by releasing a new
patch version rather than backporting to older lines.

| Crate              | Supported              |
| ------------------ | ---------------------- |
| `forge-alloc`      | latest `0.3.x`         |
| `forge-alloc-core` | latest `0.2.x`         |

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** — do not open a public
issue for an unfixed security defect.

- Preferred: GitHub private vulnerability reporting via the **Security** tab of
  <https://github.com/dmaesj/forge-alloc> ("Report a vulnerability").
- Alternative: email **dj@dominateservice.com** with a description, affected
  versions, and a reproducer if available.

You can expect an initial acknowledgement within a few business days. Once a
fix is prepared we will coordinate a release and, with your permission, credit
you in the changelog and the advisory.

## Scope

In scope: memory-safety defects (use-after-free, out-of-bounds, double-free,
uninitialized reads), unsoundness in any `unsafe` block, bypasses of a
hardening wrapper's stated guarantee, and integer/layout overflows that lead to
any of the above.

Out of scope: misuse of an `unsafe` API in violation of its documented safety
contract, denial-of-service from caller-controlled allocation sizes, and
issues that require a nightly-only or non-default build configuration the
crate does not advertise as supported.
