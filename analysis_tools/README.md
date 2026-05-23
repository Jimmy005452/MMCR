# Analysis Tools

Utility scripts for inspecting task vectors and merge behavior.

## Task vector similarity

```bash
python -m analysis_tools.task_vector_similarity --checkpoint-root checkpoints --datasets sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd --output-dir results/task_vector_similarity
```

Use `--no-plot` if you only want CSV/JSON outputs.
