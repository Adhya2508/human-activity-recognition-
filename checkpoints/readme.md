## Running the Evaluation

Run the evaluation script using:

```bash
python3 evaluate.py
```

## Saving Trained Model and Results

Stage the required files:

```bash
git add -f checkpoints/best_run2.pt
git add -f logs/history_run2.json
git add -f results/
```

Commit the changes:

```bash
git commit -m "Add Run 2 checkpoint + all evaluation results"
```

Push the changes to the remote repository:

```bash
git push
```
