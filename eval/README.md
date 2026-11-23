# HOT3D-Grasp evaluator

Per-frame metrics of generated hand-object motions (penetration ID/IV, contact, Part Acc.,
PCP, object lift along the HOT3D world up, GSR) and the physics-simulation metric.

```bash
# from the repository root
eval/run_eval_gpu.sh --device cuda:0 --input $PWD/results/graze_obj0.pkl \
  --new-metric-csv-output $PWD/results/graze_obj0.csv --new-metric-md-output $PWD/results/graze_obj0.md
(cd eval/hot3d/visualize-code && python sim_displacement.py --input $PWD/../../../results/graze_obj0.pkl \
  --workers 8 --out-csv $PWD/../../../results/sim_obj0.csv)
```

Inputs it needs (not distributed):

- `eval/data/obj.pkl` and `eval/data/object_mesh/<id>.ply`: link them to `data/hot3d/dataset/obj.pkl`
  and `data/hot3d/object_mesh` (see the main README).
- MANO models in `data/mano/mano_v1_2/models/` (or set `MANO_MODEL_DIR`).
- Part labels: `benchmark/hot3d_grasp/labels_merged.json` (default of `--part-labels`).

The GSR lift test uses the HOT3D world up `u = [0.0946, -0.6165, 0.7816]`; the column
`Object_lift_y_m` holds the lift along `u`. `scripts/make_table1.py` turns the CSVs into the
paper's table row.
