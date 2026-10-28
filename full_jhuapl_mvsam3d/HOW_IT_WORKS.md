# How it works

## The idea, next to Mehar's pipeline

Both pipelines turn the WRIVA photos into one mesh per building, placed in the scene's world frame.
Both use the same ground truth: the COLMAP camera poses and the metric depth maps. There is no true GT
depth. The depth is MoGe's, made consistent with the GT poses by a scale (`depth_scale.json`, or measured
here by `inspect`).

| | Mehar (SAM3D, single view) | here (MV-SAM3D) |
|---|---|---|
| views per building given to the model | 1 photo + its mask + its metric pointmap | up to 8 photos + masks + metric pointmaps, all of the same building |
| which pixels are "this building" | SAM 3 masks, picked per view | SAM 3 masks (or his), matched across views in 3D (`associate`) |
| from model output to world | SAM3D pose, camera frame → world with the GT pose | the same chain, from the first (reference) input view |
| up axis | mesh up aligned with the ground normal | the ground plane is fitted once for the site (`prepare`); the fit rotates about its normal |
| fit afterwards | yaw, x, y, horizontal scale, vertical scale on IoU in other views | the same five parameters on the same kind of IoU (`place`) |
| scored on | IoU in other views | IoU on test views that neither MV-SAM3D nor the fit used, plus the same numbers on all views |

The hypothesis is that seeing a building from several sides fixes what one view can't: its height, its
back side and its footprint. So the IoU should already be high before the fit, and the fit's vertical
scale `s_v` should stay near 1 instead of 0.25–0.5.

## Stages

Each stage reads the previous stage's files under `<work_root>/<run_name>/` and writes its own folder.
Stages can be rerun alone, and they skip finished work.

**`doctor`.** It doesn't produce anything. It checks that the data, GPUs, MV-SAM3D checkpoint and imports, SAM 3
and `HF_TOKEN` are there, in whichever conda env it runs in.

**`inspect`** (CPU). It checks the data:

- **Poses.** It finds the COLMAP model: the first of `sparse/sparse` (where WRIVA keeps it), `sparse` and
  `sparse/0` with `cameras.txt` + `images.txt`. `colmap_rel` overrides this. It reads the cameras (COLMAP
  pinhole, turned into OpenCV intrinsics with the principal point shifted by −0.5 px).
- **Depth files.** It finds the depth file for each image.
- **Depth scale.** It projects COLMAP's 3D points into each image and compares each point's camera depth
  with the depth map at that pixel. WRIVA ships `points3D.ply`, which doesn't say which point was seen in
  which image. Points hidden behind something show up as outliers on one side, so the scale is the most
  common ratio (its mode). With a `points3D.txt`, the observed points are used directly. The ratio is the
  factor the depth needs (1.0 = already consistent). It is reported next to `depth_scale.json`, and the
  report recommends `depth_scale_mode`: `auto` (per-image fitted scale), `json` or `none`.
- **Depth convention.** It checks whether the files hold z-depth or distance along the ray. MVE-style
  depth maps can be either. If the files were distances along the ray, the ratio above would shrink
  toward the image edges like cos(angle off the optical axis). The report fits that trend: slope 0 means
  z-depth, slope 1 means along the ray. `depth_kind=ray` converts.

**`prepare`** (CPU). It stores the per-image depth scales, then lifts depth from all views into a world
point cloud. A RANSAC plane fit to that cloud gives the ground (in an aerial scene, the largest flat
surface) and the up direction (the side the cameras are on). It also writes the site extent, an automatic
voxel size (extent / 400) and the background point cloud used in the final GLB.

**`masks`** (GPU, SAM 3) or **`import_masks`**. These give per-view instance masks for each text prompt
(`building`): every building SAM 3 finds in that photo, stored bit-packed in `masks/<prompt>/<image>.npz`.
`import_masks` writes the same files from Mehar's masks: label maps, one image per building, or `.npz`/`.npy` stacks.

**`associate`** (CPU). This decides which mask in which photo belongs to which building, using geometry
rather than appearance:

1. Every mask pixel (every `lift_stride`-th) is lifted to 3D with its GT depth and pose.
2. The 3D samples are dropped into voxels. A voxel counts if masks from at least `min_view_support` views
   land in it, which removes depth noise and one-off false positives.
3. The surviving voxels are split into connected components (26-neighbourhood). Each component is one
   building candidate; it is kept if it is big enough and seen by at least `min_object_views` views.
4. Each mask joins the building where most of its 3D samples land (at least `assign_min_frac`). SAM sometimes merges two buildings
   in one photo (one behind the other). If a second building, confirmed by other views, holds at least
   `split_min_frac` of the mask, the mask is cut pixel by pixel by nearest building voxel, at full depth
   resolution.
5. Views are chosen per building for MV-SAM3D. Each view is scored by coverage (share of the building's
   voxels that project inside its mask) × √(mask area), halved if the mask touches the image border (the
   building is cut off). Views are then picked greedily, rewarding new viewing directions. The best
   un-truncated view goes first and becomes the reference view.

The voxel size adapts: at least 2.5 × the spacing of the lifted samples, so that thin or distant
buildings don't fall apart into separate components.

**`mvsam3d`** (GPU). For each building, biggest first, the stage writes the input MV-SAM3D expects into
`objects/<hash>_<building>/`:

- the photos;
- RGBA images whose alpha channel is this building's mask (other buildings are not in it);
- `da3_output.npz`, where MV-SAM3D normally expects a DepthAnything-3 prediction. Here it holds
  pointmaps built from the GT depth (camera-space OpenCV XYZ: x right, y down, z forward, in metres), the
  GT world→camera extrinsics and the intrinsics.

Then `MV-SAM3D/run_inference_weighted.py` runs as a subprocess, with the working directory at the MV-SAM3D
root (its scripts rely on that), one GPU (`CUDA_VISIBLE_DEVICES`) and a timeout. Its output folder
(`result.glb`, `params.npz`) is moved into the building folder. Because the pointmaps are metric and come
from the GT poses, the pose MV-SAM3D returns is metric too, in the reference camera's frame.

**`place`** (GPU, small). This puts each mesh into the world and scores it:

```
canonical mesh (z-up) ─Z_UP_TO_Y_UP→ y-up ─(× scale) · R + t→ PyTorch3D camera ─diag(−1,−1,1)→ OpenCV camera ─c2w(reference view)→ world
```

`placed.glb` is that result, with no fitting. It is scored in every view where the building has a mask
(next section). Then the fit (`refine`) adjusts yaw about the ground normal, x/y along the ground,
horizontal scale and vertical scale about the building's base centre, to maximize silhouette IoU. It
starts with a coarse yaw scan (every 15°) and a vertical-scale scan (2^−1.5 … 2^1.5), then runs
Nelder–Mead, with scales bounded to [1/4, 4]. The result is `refined.glb`.

**`assemble`** (CPU). It merges all buildings into `scene/scene_objects_{placed,refined}.glb` (plus the
background point cloud), and writes `objects.csv` and `summary.json`.

## Scoring: silhouette IoU

A building's mesh is sampled densely (enough points to cover its largest mask) and projected into a view.
That gives a silhouette, compared with the building's mask in that view (intersection over union). Two
details keep the number honest:

- **Occlusion.** A projected point that lies more than 5% (`occlusion_tol`) behind the photo's GT depth at
  that pixel is hidden by something in front, e.g. another building or a tree. It is removed. Otherwise a
  correct mesh would be penalized for pixels the mask cannot contain.
- **Gaps.** A point splat has holes between points; a small morphological closing fills them. A plain
  dilation would make every outline fat and lower the IoU of a perfect mesh.

Views are split three ways per building:

- **input**: the ≤ 8 views MV-SAM3D saw.
- **held-out**: all other views of the building.
- **test**: every third held-out view. Neither MV-SAM3D nor the fit uses them.

The fit chooses its ≤ 12 views (largest masks) from input + held-out, minus test. So
`iou_test_refined` measures whether the fit generalizes, not just whether it can match the views it was fitted on.

## Coordinate conventions

- **COLMAP** `images.txt` stores world→camera (qvec wxyz, tvec) in the OpenCV camera convention (x right,
  y down, z forward). Intrinsics come from `cameras.txt` (pinhole models; undistorted images). The pixel
  centre convention differs by 0.5 px from OpenCV's, which is why −0.5 is applied to the principal point.
- **Depth files** should be z-depth (distance along the optical axis). If `inspect` finds they hold the
  distance along the ray, `depth_kind=ray` converts them (z = distance × cos(angle off the axis)). They
  may be at a lower resolution than the photo; intrinsics are rescaled to the depth resolution. The
  geometry uses COLMAP's intrinsics, not the `_intrinsics.json` MoGe estimated; `inspect` prints both.
- **MV-SAM3D / SAM3D** poses are in PyTorch3D's camera convention (x left, y up, z forward), with
  row-vector rotations (`p @ R`). The canonical object frame is z-up. `scene_mvsam3d/placement.py` holds
  the whole chain; the tests check it by inverting it exactly.

## Multiple GPUs: torchrun yes, DDP wrapper no

The DDP example (MapAnything's `train_tools.py`) does three things:

1. It starts one process per GPU (`torchrun --nproc_per_node=N`). Each process joins a process group
   (`init_process_group(backend="nccl", init_method="env://")`) and learns its `rank`.
2. It wraps the model in `DistributedDataParallel`. Every process holds a full copy; after each backward
   pass, the gradients are averaged across processes (an NCCL all-reduce), so the copies stay identical.
   A `DistributedSampler` gives each process a different slice of the data.
3. Rank 0 alone prints, saves checkpoints and averages metrics (`all_reduce_mean`); barriers keep the
   processes in step.

Here nothing is trained. Each building is an independent MV-SAM3D inference that takes minutes, so there
are no gradients to average and the DDP wrapper (step 2) would do nothing. Steps 1 and 3 are what we
need, and `scene_mvsam3d/dist.py` follows the same pattern and names:

| MapAnything example | here |
|---|---|
| `torchrun --nproc_per_node=N train.py` | `torchrun --nnodes=1 --nproc_per_node=N --master_addr=127.0.0.1 --master_port=<free> run.py {masks,mvsam3d,place}` (the launcher does it; a free port, so it can't clash with other users' jobs on the default 29500) |
| `init_distributed_mode(args)` → `init_process_group("nccl", "env://")` | `init_distributed_mode()` → `init_process_group("gloo", "env://")`, 24 h timeout |
| `setup_for_distributed(rank == 0)`, `print(..., force=True)` | the same; per-building progress is printed from every rank |
| `get_rank()`, `get_world_size()`, `is_main_process()`, `barrier()` | the same |
| `all_reduce_mean(loss)` | `all_gather_object(results)`: rank 0 writes `<stage>_summary.json` with every rank's finished and failed buildings |
| `DistributedSampler` + `set_epoch` | `claim()`: work queue in a shared folder (below) |
| `DistributedDataParallel(model)`, `no_sync`, checkpointing | not needed (no training) |

There are two deliberate differences:

- **gloo instead of NCCL.** The ranks only exchange small Python lists at the end. The GPU belongs to the
  MV-SAM3D subprocess, not to the coordinating process. gloo runs on the CPU, so it doesn't compete for
  GPU memory. The long timeout is because the final gather waits for the slowest building (up to
  `object_timeout_min`).
- **A claim queue instead of a static split.** A `DistributedSampler` would hand rank k buildings k, k+N,
  k+2N, … in advance. Buildings vary several-fold in cost (size, number of views), so some GPUs would
  finish early and sit idle. Instead, each rank walks the list (biggest first) and claims a building by
  creating `objects/<building>/.claim_mvsam3d` atomically (`O_CREAT | O_EXCL`): whoever creates the file
  first runs it. A GPU that finishes takes the next unclaimed building. Finished buildings have a result
  file and are skipped, so a crashed or interrupted run resumes where it stopped. A stale claim from a
  killed run is cleared by the launcher, or expires after the timeout.

The `masks` stage uses the simple static split (image k, k+N, …), because every image costs about the same.

**When we would need the real DDP wrapper:** if we fine-tune MV-SAM3D on WRIVA (like the Stage-2 LoRA
experiments). Then the MapAnything template applies as is: NCCL, `DistributedDataParallel`,
`DistributedSampler`, rank-0 checkpoints.

**One node is enough.** A Trinity node has 8 GPUs. The work folder is on that node's `/scratch`, so ranks
on a second node couldn't see the claim files. If we ever need two nodes, run two launchers with
different `RUN_NAME`s on disjoint subsets of buildings rather than `--nnodes 2`.

**Why MV-SAM3D runs as a subprocess:**

- its scripts expect to run from the MV-SAM3D root, with their own `sys.path`;
- a crash or out-of-memory on one building can't take the rank down;
- the timeout is easy to enforce.

The cost is reloading the model for each building. That's small next to the run itself.

## Known limitations and open questions

- **Buildings only.** Trees, cars and the ground are not reconstructed as objects. The ground and
  everything else appear only as the depth point cloud in `scene_with_background.glb`. More prompts
  (`prompts=building,tree`) work the same way.
- **Masks bound the result.** The association can fix merged instances and drop one-off false positives,
  but a building SAM never segments is missing.
- **IoU measures the silhouette, not the surface.** A building with the right outline but a wrong roof
  scores well. A geometric check against the depth maps (e.g. median depth error inside the mask) would
  be the next metric to add.
- **The reference view sets MV-SAM3D's pose frame.** Errors in that view's depth propagate to the
  placement. `mvsam3d_pose_optimization=true` lets MV-SAM3D refine its pose against all input views.
