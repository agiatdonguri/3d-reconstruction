# Full JHU APL scene with MV-SAM3D

Reconstruct every building of the full JHU APL WRIVA site with **MV-SAM3D** (multi-view SAM 3D), using the
GT metric depth and GT camera poses, and compare against Mehar's single-view **SAM3D** pipeline on the same
scene. Mehar gives SAM3D one photo per building and then fixes yaw, position and scale by fitting silhouettes
in other views (mean IoU 0.6–0.8 before that fit, 0.85–0.95 after; his meshes came out 2–4× too tall). Here
each building gets up to 8 views at once, so the shape and height should already agree with the other views.
The same silhouette fit runs afterwards, so both pipelines are compared before and after it.

How it works, stage by stage, and why it runs on several GPUs without the DDP wrapper: [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

| file | what |
|---|---|
| `run.py` | entry point for every stage (`python run.py <stage> --config … --scene …`) |
| `run_trinity.sh` | launcher on a Trinity node: copies inputs to /scratch, picks idle GPUs, runs stages, syncs results to NAS |
| `copy_from_mehar.sh` | one-time copy of the scene (and optionally Mehar's masks / results) to `/data3/tnishida` |
| `setup_sam3_env.sh` | conda env `sam3` for the SAM 3 `masks` stage (not needed with Mehar's masks) |
| `configs/jhuapl_full.yaml` | settings for this scene (every key explained in `scene_mvsam3d/config.py`) |
| `scene_mvsam3d/` | the code |
| `tests/` | CPU tests on a synthetic two-building scene (`python -m pytest tests -q`) |

## 0. The data, and what is still open with Mehar

Each WRIVA scene on Trinity is one folder `/data2/mehark/WRIVA/<scene>/`, e.g.
`t01_v01_s06_r08_ImageDensity_A01_iPad_LTS`. This is the layout Mehar described, and what the pipeline
uses from it:

| in the scene folder | what | used here |
|---|---|---|
| `sparse/undistorted_mitigated_images/0/` | the photos | yes |
| `sparse/sparse/cameras.txt`, `images.txt` | COLMAP cameras and poses (the GT poses) | yes |
| `sparse/sparse/points3D.ply` | COLMAP 3D points | yes, to check the depth scale |
| `sparse/metric_z_depth_mve/image_NNNN_zdepth.npy` (+ `_intrinsics.json`) | MoGe depth per photo | yes, as the depth |
| `sparse/depth_scale.json` | metric scale factor | `inspect` compares it with its own measurement |
| `sparse/sparse_nvm/sparse.nvm` | the same poses in NVM format | no |
| `sparse/gsplat/` (+ `world_normalization_transform.npy`) | Gaussian splatting assets | no |
| `sparse/map_anything_rescaled_sparse/` | MapAnything reconstruction with a separate camera set | no (not copied) |
| `model/` | the Accenture site model (`*_acc_mesh.ply`, `*_acc_splat.ply`, …) | reference only |
| `metrics/` | reference images and metric outputs | reference only |

There is no true ground-truth depth. The "GT depth" is the MoGe depth made metric, which is what
Mehar's pipeline uses too. His description is two months old, so `inspect` (step 1) re-checks all of
it on the real folder:

- which COLMAP folder it found (it tries `sparse/sparse`, then `sparse`, then `sparse/0`);
- how many photos have depth;
- which depth scale makes the depth agree with the poses;
- whether the depth files are z-depth or distance along the ray.

What is still open. A message you can send:

> Hi Mehar, for the MV-SAM3D run on JHU APL could you send me:
> 1. the scene folder for the full JHU APL site: one folder (one COLMAP model), or several captures?
> 2. do the `_zdepth.npy` files already include `depth_scale.json`, or do you multiply by it?
> 3. your SAM 3 building masks for that scene (path + format), so we both use the same masks;
> 4. your per-building IoUs before and after alignment, which views you computed them on (the views you
>    optimized on, or others?), and where your aligned SAM3D meshes are;
> 5. if the scene mixes drone and ground (iPad) photos, a filename pattern for the drone ones.

- **Point 1:** the pipeline takes one scene folder. Several captures only work together if they share
  one COLMAP frame.
- **Point 2:** `inspect` measures this anyway; his answer is a cross-check.
- **Point 3:** optional. Without his masks we run SAM 3 ourselves, but the comparison is cleaner with the
  same masks.
- **Point 4:** needed for a fair comparison (section 4).

## 1. One-time setup on Trinity

Trinity has no SLURM: pick a node with free GPUs on the [monitor](http://trinity.vision.cs.cmu.edu:53240/)
(campus/VPN, or `ssh -L 127.0.0.1:53240:localhost:53240 tnishida@trinity.vision.cs.cmu.edu`), and do
everything below on that node, not on the head node.

```bash
ssh trinity-X-Y
cd ~/3d-reconstruction && git pull            # after you commit + push this folder
cd full_jhuapl_mvsam3d
export SCENE=<scene folder name from Mehar>
```

**Data.** Copy the scene to your NAS once; runs never read Mehar's tree directly. The script shows the
sizes and asks before copying. It copies what the table in section 0 marks as used. Two options:

- `COPY_EVAL=1` also copies `model/` (the Accenture site model) and `metrics/`, for comparison only.
- `MASKS_SRC=<path>` copies his masks.

```bash
SCENE=$SCENE MASKS_SRC=<his masks folder> COPY_EVAL=1 ./copy_from_mehar.sh
# → /data3/tnishida/WRIVA/$SCENE, …/${SCENE}__masks, …/${SCENE}__mehar_eval
```

**Conda env `mvsam3d`** (MV-SAM3D; runs every stage except `masks`). If it doesn't exist yet:
`../scripts/setup_conda_trinity.sh`, then the heavy SAM 3D stack as in [`../docs/SETUP_ENV.md`](../docs/SETUP_ENV.md).
Then the SAM 3D checkpoint, the same way the Colab notebooks do it (needs access to
[facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)):

```bash
conda activate mvsam3d
export HF_TOKEN=hf_…        # type it in the shell; never put it in a file
python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download('facebook/sam-3d-objects', token=os.environ['HF_TOKEN'], local_dir=os.path.expanduser('~/code/sam-3d-objects/checkpoints/hf'))"
CK=~/code/sam-3d-objects/checkpoints/hf; [[ -f $CK/checkpoints/pipeline.yaml ]] && CK=$CK/checkpoints
mkdir -p ~/code/MV-SAM3D/checkpoints && ln -sfn "$CK" ~/code/MV-SAM3D/checkpoints/hf
ls ~/code/MV-SAM3D/checkpoints/hf/pipeline.yaml
pip install trimesh pyyaml matplotlib
```

**Conda env `sam3`** (only if you run SAM 3 yourself): `./setup_sam3_env.sh` (needs access to
[facebook/sam3](https://huggingface.co/facebook/sam3)).

**Check everything:** prints ✓/✗ per item, in both envs (data, GPUs, MV-SAM3D checkpoint and imports, SAM 3, HF_TOKEN):

```bash
STAGES=doctor ./run_trinity.sh
```

Each line says which stage needs it. Two ✗ are expected: MV-SAM3D in the `sam3` env, and SAM 3 in the
`mvsam3d` env.

## 2. Run

Use tmux so an ssh drop doesn't kill the run (`Ctrl-b d` detaches, `tmux attach -t jhuapl` returns):

```bash
tmux new -s jhuapl
cd ~/3d-reconstruction/full_jhuapl_mvsam3d
export HF_TOKEN=hf_… SCENE=<scene>
```

The launcher uses the GPUs that are idle right now (`GPUS=0,1,2,3` to choose), writes to
`/scratch/tnishida/outputs/full_jhuapl_mvsam3d/<scene>/`, and copies that folder to
`/data3/tnishida/outputs/full_jhuapl_mvsam3d/<scene>/` after every stage. Every stage skips work that is
already done, so rerunning a command resumes it.

**Step 1: check the data** (minutes, CPU).

```bash
STAGES=inspect ./run_trinity.sh
```

Read `inspect/report.json` (also printed to the console):

- `colmap_dir`: where it found the poses (expected: `…/sparse/sparse`); `points3d`: the point file it used.
- `n_registered_images` and `n_with_depth`: how many photos have poses, and how many of those have depth.
- `calibration_vs_colmap` and `recommendation`: the depth scale. If it recommends `json` or `none`, add
  `EXTRA_SETS="depth_scale_mode=json"` (or `none`) to every command below, or put
  `depth_scale_mode: json` in the config.
- `depth_convention.verdict`: should start with `ok`. If it says `set depth_kind=ray`, the files hold
  distance along the ray rather than z-depth. Add `depth_kind=ray` the same way and rerun `inspect`
  until the verdict says ok.
- `depth_examples`: per photo, MoGe's own intrinsics (`intrinsics_json`) next to COLMAP's focal length at
  the depth resolution. They should be close.

**Step 2: ground plane, masks, buildings** (SAM 3 takes about 1–2 s per image per GPU; association is CPU).

```bash
STAGES=prepare,masks,associate ./run_trinity.sh
# or with Mehar's masks (copied during setup):
STAGES=prepare,import_masks,associate MASKS_IMPORT_DIR=/data3/tnishida/WRIVA/${SCENE}__masks ./run_trinity.sh
```

If his masks are one file per building (e.g. `image_0001_3.png`), add `EXTRA_SETS="masks_import_pattern={stem}_*"`.
Then check three things:

- `prepare/scene.json`: `camera_height_above_ground_median` should be about the drone altitude in metres.
  If not, the ground plane latched onto something else, e.g. a big flat roof.
- `associate/objects_topdown.png`: one colour per building, seen from above.
- `associate/view_*.png`: the buildings drawn on the photos.

You should see one colour per building, no building split into pieces, and no two buildings sharing a
colour. Section 6 lists the knobs if not.

**Step 3: smoke test on the 3 biggest buildings** (roughly 15–30 min).

```bash
STAGES=mvsam3d,place,assemble EXTRA_SETS="max_objects=3" ./run_trinity.sh
```

Open `scene/scene_objects_refined.glb` (section 3) and check the timing: each `objects/*/mvsam3d_result.json`
has `seconds`. If a building failed, its `mvsam3d.log` says why.

**Step 4: all buildings.**

```bash
STAGES=mvsam3d,place,assemble ./run_trinity.sh
```

The three buildings from step 3 are skipped. The total is about (number of buildings × minutes per
building) ÷ GPUs, e.g. 100 buildings × 10 min ÷ 8 GPUs ≈ 2 h. Use the step-3 timing for your estimate.

Without the launcher (any machine with the envs), use
`python run.py <stage> --config configs/jhuapl_full.yaml --scene $SCENE --set key=value …`. For the GPU
stages, use the same torchrun flags as the launcher:

```bash
torchrun --nnodes=1 --nproc_per_node=<gpus> --master_addr=127.0.0.1 --master_port=<free port> \
  run.py {masks|mvsam3d|place} --config configs/jhuapl_full.yaml --scene $SCENE
```

## 3. Outputs

Everything is under `/data3/tnishida/outputs/full_jhuapl_mvsam3d/<scene>/`:

| path | what |
|---|---|
| `scene/summary.json` | the headline numbers (section 4) |
| `scene/objects.csv` | one row per building: status, input views, IoUs, fitted scales and yaw |
| `scene/scene_objects_refined.glb` | every building after the silhouette fit, in world coordinates |
| `scene/scene_objects_placed.glb` | every building at MV-SAM3D's own pose (no fit) |
| `scene/scene_with_background.glb` | refined buildings + the depth point cloud of the rest of the scene |
| `associate/objects_topdown.png`, `view_*.png` | which pixels belong to which building |
| `objects/<hash>_<building>/` | per building: `images/` + `<building>/` (MV-SAM3D input: photos, RGBA masks, `da3_output.npz` with GT pointmaps and poses), `mvsam3d/` (raw MV-SAM3D output), `mvsam3d.log`, `placed.glb`, `refined.glb`, `place_result.json` (IoU per view) |
| `inspect/report.json`, `prepare/scene.json` | data check; ground plane, up axis, scene size |
| `masks_summary.json`, `mvsam3d_summary.json`, `place_summary.json` | which buildings each GPU stage finished or failed |
| `logs/run_*.log` | the full console output of each launcher run |

To view on your laptop, copy the `scene/` folder, e.g.
`rsync -av tnishida@trinity.vision.cs.cmu.edu:/data3/tnishida/outputs/full_jhuapl_mvsam3d/<scene>/scene/ ./jhuapl_scene/`.
Then open the `.glb` files in MeshLab (File → Import Mesh) or Blender (File → Import → glTF). The GLBs
keep MV-SAM3D's textures. Buildings are separate nodes named after their ids, so you can find one from
`objects.csv`.

## 4. What to report, and comparing fairly with Mehar

`scene/summary.json` has mean IoUs over all buildings. A building is scored in one of three view sets:

- **test**: views MV-SAM3D never saw and the silhouette fit never used. This is the fair number.
- **heldout**: all views MV-SAM3D never saw. The fit may have used some of them.
- **all**: every view that sees the building.

Each IoU comes in two versions. **placed** is MV-SAM3D's own pose, with nothing fitted. **refined** is
after the yaw / x / y / scale fit (Mehar's parametrization).

- `mean_iou_test_placed` vs Mehar's **before alignment** (0.6–0.8). This is the main multi-view claim:
  MV-SAM3D output placed with the GT depth and poses, with no fitting at all.
- `mean_iou_test_refined` vs his **after alignment** (0.85–0.95). Check with him which views his numbers
  used. If he measured on the views he optimized on, compare with our `iou_all_refined` as well.
- `scale_v` in `objects.csv`: the height correction the fit needed. His SAM3D meshes needed about
  0.25–0.5 (2–4× too tall). Values near 1 mean multi-view fixed the height.
- Figures:
  - `associate/objects_topdown.png`;
  - a screenshot of `scene_with_background.glb`;
  - a few buildings side by side with his aligned SAM3D meshes (ask him where they are);
  - optionally, against the Accenture site mesh (`${SCENE}__mehar_eval/model/*_acc_mesh.ply`, copied with `COPY_EVAL=1`).

Same masks (his) and same views make the comparison cleanest. Use `RUN_NAME=<name>` to keep variants
apart, e.g. `RUN_NAME=${SCENE}_mehar_masks`.

## 5. Settings you might change

Pass these as `EXTRA_SETS="key=value key2=value2"`, or edit `configs/jhuapl_full.yaml`. All keys are
explained in `scene_mvsam3d/config.py`.

| key | default | when to change |
|---|---|---|
| `max_objects` | 0 (all) | smoke tests |
| `depth_scale_mode` | `auto` | what `inspect` recommends (`json` / `none`) |
| `depth_kind` | `z` | `ray` if `inspect` says the files hold distance along the ray |
| `colmap_rel` | auto | the folder with `cameras.txt` + `images.txt`, if it is not `sparse/sparse`, `sparse` or `sparse/0` |
| `views_per_object` | 8 | fewer if MV-SAM3D runs out of memory or is slow |
| `image_max_side` | 1600 | lower (1200) for memory |
| `image_glob` | `*` | e.g. `*drone*` to use drone images only |
| `prompts` | `[building]` | `building,tree` etc. for more object types (SAM 3 masks) |
| `refine_dofs` | `yaw,tx,ty,s_h,s_v` | add `tz` to let the fit move buildings up/down |
| `mvsam3d_pose_optimization` | false | MV-SAM3D's own pose optimization before placement |
| `stage1_steps`, `stage2_steps` | 50, 25 | fewer for quick tests |

## 6. Troubleshooting

- **"no idle GPU"**: another user is on every GPU. Pick another node, or set `GPUS=…` yourself.
- **"no cameras.txt + images.txt"**: the poses are elsewhere. Find them with
  `find /data3/tnishida/WRIVA/$SCENE -name images.txt`, then set `colmap_rel` to that folder (relative to
  the scene). If only `.bin` files exist, convert them to text in your copy (the error message gives the command).
- **torchrun hangs or fails at startup** (`DistNetworkError`, "client socket has timed out"): the ranks
  can't find each other. The launcher already uses `127.0.0.1` and a free port. If you run torchrun by
  hand, use the same flags (section 2), not `--standalone`.
- **doctor ✗ MV-SAM3D imports**: the `mvsam3d` env is incomplete (kaolin / spconv / pytorch3d …). See
  `../docs/SETUP_ENV.md`; the full error is `python -c "import sys; sys.path.append('notebook'); from inference import Inference"` run inside `~/code/MV-SAM3D`.
- **A building failed in mvsam3d**: read `objects/<…>/mvsam3d.log`.
  - CUDA out of memory: lower `views_per_object` or `image_max_side`.
  - Timeout: raise `object_timeout_min`.
  - Rerunning `STAGES=mvsam3d` retries only the failed buildings.
- **Association problems** (from the step-2 figures):
  - One building in several pieces: raise `voxel_size` (metres; `prepare/scene.json` has the automatic value).
  - Two buildings merged in a photo: lower `split_min_frac` (0.05).
  - Buildings missing: lower `min_object_views` (3) or `mask_min_area_frac`.
- **Re-doing a stage**: delete its outputs first.
  - Redo place: delete `objects/*/place_result.json`.
  - Redo association: delete `associate/` and `objects/`. The building ids change, so a new `RUN_NAME` is cleaner.
- **Depth looks wrong** (buildings floating or sunk in `scene_with_background.glb`, or flat buildings
  bent near the photo edges): check `inspect/report.json`, `depth_scale_mode` and `depth_kind`.

## 7. Tests

On any machine with numpy/scipy/torch/trimesh (CPU, ~10 s), from this folder:

```bash
python -m pytest tests -q
```

The tests use a synthetic scene with two box buildings and 8 aerial cameras, laid out like a WRIVA scene
folder (section 0). They cover:

- finding the COLMAP model in `sparse/sparse` and ignoring the MapAnything camera set;
- the depth scale, from `points3D.ply` and from COLMAP tracks;
- detecting and converting depth stored as distance along the ray;
- the ground plane;
- association, including splitting two buildings that SAM merged in one view;
- the MV-SAM3D input pointmaps;
- exact placement;
- the fit of a 2.5× too tall, 12°-rotated mesh;
- mask import;
- the MV-SAM3D stage under real `torchrun` with 2 ranks, using a stand-in for MV-SAM3D.
