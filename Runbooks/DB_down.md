# Runbook: Postgres Unavailable

Symptoms:
- `/health` shows postgres: DOWN
- API returns 503 or cannot create records
- Prometheus alerts for DB latency or connection errors

Immediate Actions:
1. Check Postgres container logs:

   docker compose logs postgres

2. Check Postgres process and disk usage on host. If out of disk, free up space or increase volume.

3. Verify network connectivity from API/worker containers:

   docker compose exec api ping -c 3 postgres

4. If Postgres container is crashed, try restarting:

   docker compose restart postgres

5. If data corruption is suspected, consult backups and failover plan. Restore from backup to a new instance and reconfigure the application to point to the new DB.

6. After remediation, verify health endpoint and run smoke tests.

Post-mortem:
- Record root cause, timeline, mitigation, and actions to prevent recurrence (monitoring, backups, resource limits).
