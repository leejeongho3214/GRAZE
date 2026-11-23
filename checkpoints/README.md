# Checkpoints

One GRAZE checkpoint per object fold (`graze_obj0.pth`, `graze_obj1.pth`, `graze_obj2.pth`),
attached to the [v1.0 release](https://github.com/leejeongho3214/GRAZE/releases/tag/v1.0).
Download them into this folder and verify them:

```bash
for k in 0 1 2; do
  wget -P checkpoints https://github.com/leejeongho3214/GRAZE/releases/download/v1.0/graze_obj${k}.pth
done
(cd checkpoints && sha256sum -c SHA256SUMS)
```

## License

The checkpoints are released under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
They were trained on [HOT3D](https://facebookresearch.github.io/hot3d/) and work only with the BPS
basis of [DiffH2O](https://github.com/facebookresearch/diffh2o) (CC BY-NC 4.0), so they may not be
used for commercial purposes. The code that loads them is MIT-licensed (see `../LICENSE`).
