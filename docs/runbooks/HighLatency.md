# High API Latency Runbook

## Alert Details
- **Alert Name**: HighAPILatency
- **Severity**: Warning
- **Description**: 95th percentile API latency is above 500ms for 5 minutes
- **Threshold**: P95 latency > 500ms

## Impact
- Degraded user experience
- Potential timeout issues for client applications
- Possible queue buildup of XAI tasks

## Possible Causes
1. Database performance issues
2. MLflow model loading delays
3. Resource contention (CPU/Memory)
4. Network latency between services
5. High concurrency leading to worker pool exhaustion

## Investigation Steps

### 1. Check System Resources
```bash
# Check API pod resource usage
kubectl top pod -l component=api

# Check node metrics
kubectl top nodes
```

### 2. Check Database Performance
```bash
# Check Postgres metrics in Grafana
# Look for:
- Connection count
- Query latency
- Lock wait time
```

### 3. Verify MLflow Connectivity
```bash
# Check MLflow pod status
kubectl get pods -l app=mlflow

# Check MLflow logs
kubectl logs -l app=mlflow
```

### 4. Analyze Request Patterns
1. Check Grafana dashboards for:
   - Request rate
   - Endpoint distribution
   - Error rates

2. Check for correlation with specific endpoints or request patterns

## Resolution Steps

### If Database-Related
1. Check for long-running queries:
```sql
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE pg_stat_activity.query != '<IDLE>' 
AND pg_stat_activity.query NOT ILIKE '%pg_stat_activity%' 
ORDER BY duration DESC;
```

2. Consider connection pooling adjustments in `api/app.py`

### If Resource-Related
1. Scale up API replicas:
```bash
kubectl scale deployment fraud-api --replicas=3
```

2. Consider updating resource limits in Helm values

### If MLflow-Related
1. Verify model artifacts are properly cached
2. Check model loading logic in API startup
3. Consider implementing model caching improvements

## Prevention
1. Set up autoscaling based on CPU/Memory
2. Implement request rate limiting
3. Cache frequently accessed model artifacts
4. Monitor trends in latency metrics
5. Regular database maintenance

## Escalation
If unable to resolve:
1. Escalate to database team if DB-related
2. Escalate to MLOps team if model-related
3. Escalate to platform team if infrastructure-related

## Related Resources
- Grafana Dashboard: http://grafana:3000/d/fraud-api/
- Architecture Diagram: /docs/architecture.md
- Scaling Guidelines: /docs/scaling.md