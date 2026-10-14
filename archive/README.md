# Historical references

These files preserve earlier experiments and plans. They are not used by the API, container build, or current CI. Their imports, dataset assumptions, deployment images, and recovery steps have not been updated or verified.

- `experiments/`: earlier data loading, EDA, model explanation, and Flask experiments.
- `k8s/`: older raw Kubernetes manifests. Use the current Helm chart as the maintained deployment example.
- `data/credit_approval.csv`: the original 690-row credit-approval sample, not the 30-feature fraud dataset.
- `project-roadmap.md` and `worker-recovery-test-plan.md`: original planning references, preserved without rewriting their contents.
