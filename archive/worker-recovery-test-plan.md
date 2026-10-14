# Worker Recovery Test Plan

## Overview
This test plan validates the auto-recovery capabilities of the Celery workers in the fraud detection system, particularly focusing on task reprocessing after worker failure.

## Prerequisites
- Kubernetes cluster with KEDA installed
- Helm chart deployed with fraud-detection stack
- kubectl access to the cluster
- curl or similar HTTP client for API testing

## Test Cases

### 1. Basic Worker Recovery

#### Setup
1. Ensure the system is running:
```bash
kubectl get pods -l app=fraud-detection
```

2. Start monitoring logs:
```bash
kubectl logs -f -l component=xai-worker
```

#### Test Steps
1. Submit multiple prediction requests:
```bash
for i in {1..10}; do
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
done
```

2. Verify tasks are queued:
```bash
kubectl exec deploy/fraud-api -- celery -A xai_tasks inspect active
```

3. Simulate worker failure:
```bash
kubectl delete pod -l component=xai-worker
```

4. Verify recovery:
- Watch for new worker pod creation
- Check task reprocessing in logs
- Verify results in database

#### Expected Results
- New worker pod starts automatically
- Tasks are reprocessed (due to acks_late=True)
- All SHAP explanations are eventually computed
- No data loss occurs

### 2. KEDA Scaling Test

#### Test Steps
1. Generate high load:
```bash
for i in {1..50}; do
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
done
```

2. Monitor scaling:
```bash
watch kubectl get pods -l component=xai-worker
```

#### Expected Results
- KEDA scales workers up when queue length > 5
- Workers scale down after queue clears
- Eventually scales to zero if no tasks

### 3. Graceful Shutdown Test

#### Test Steps
1. Submit a few long-running tasks
2. Scale down the deployment:
```bash
kubectl scale deployment xai-worker --replicas=0
```

#### Expected Results
- Workers complete in-progress tasks before shutting down
- No tasks are lost during scale-down

## Monitoring During Tests

1. Watch Prometheus metrics:
- Queue length
- Task processing time
- Error rates

2. Check Grafana dashboards:
- Worker pod count
- Task throughput
- Recovery times

3. Verify data consistency:
- All transaction_ids have corresponding SHAP values
- No duplicate computations
- Status field reflects final state

## Recovery Validation Queries

Check task completion in PostgreSQL:
```sql
SELECT 
    COUNT(*) as total_requests,
    SUM(CASE WHEN shap_values IS NOT NULL THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN shap_values IS NULL THEN 1 ELSE 0 END) as pending
FROM transaction_results;
```

## Success Criteria
- 100% task completion after recovery
- No data loss or corruption
- Proper scaling behavior
- Graceful shutdown works as expected