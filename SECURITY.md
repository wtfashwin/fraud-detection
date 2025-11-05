Security Hardening Checklist
---------------------------

1. Secrets & configuration
   - Do NOT store production secrets in `.env` committed to the repo.
   - Use platform secrets (Kubernetes Secrets, Docker secrets, or HashiCorp Vault).

2. Image hardening
   - Build images as a non-root user (Dockerfile updated).
   - Keep base images up-to-date and use minimal variants.

3. Dependency scanning
   - Add `trivy` or `snyk` to CI to scan images and dependencies.

4. Runtime security
   - Use pod security policies, resource limits, and read-only filesystems when possible.
