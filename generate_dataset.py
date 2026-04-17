#!/usr/bin/env python3
"""
High-quality & optimized per-scene dataset generator for Habitat/ScanNet.

Usage:
  # Driver mode: spawn a process per scene (resumable)
  python3 generate_dataset.py --driver --start-index 0

  # Worker mode: process a single scene (invoked by driver or manually)
  python3 generate_dataset.py --scene <scene_name> --top-k 5 --save-size 256 256
"""
import argparse
import os
import sys
import subprocess
import json
import gzip
import yaml
import logging
import random
import time
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")


# Config & defaults
DEFAULT_OUTPUT_DIR = "dataset_root"
DEFAULT_GOAT_BENCH = "data/datasets/goat_bench/hm3d/v1"
DEFAULT_SPLIT = "train"
DEFAULT_TOP_K = 100
DEFAULT_SAVE_SIZE = (256, 256)
CONFIG_YAML = "config.yaml"
AGENT_HEIGHT = 1.5


# Helper wrapper for the pool to call
def spawn_worker_process(task_args):
    """
    Independent function to be pickled by ProcessPoolExecutor.
    """
    scene_name, goat_path, split, out_root, top_k, save_size = task_args

    # Check if done already to save process startup time
    scene_out = Path(out_root) / "rgb" / scene_name
    if scene_out.exists() and any(scene_out.iterdir()):
        print(f"[driver] skipping {scene_name} (already present)")
        return

    print(f"[driver] spawning worker for scene: {scene_name}")
    
    cmd = [
        sys.executable,
        os.path.realpath(__file__), # Points to this script itself
        "--scene", scene_name,
        "--goat-bench", goat_path,
        "--split", split,
        "--output-dir", str(out_root),
        "--top-k", str(top_k),
        "--save-size", str(save_size[0]), str(save_size[1]),
    ]

    try:
        # We use subprocess here so each worker has its own distinct Python memory space
        # This isolates memory leaks to a single process which dies when the scene ends.
        start = time.time()
        ret = subprocess.call(cmd)
        duration = time.time() - start
        print(f"[driver] scene {scene_name} done (exit {ret}) in {duration:.1f}s")
    except Exception as e:
        print(f"[driver] failed to run worker for {scene_name}: {e}", file=sys.stderr)


def run_driver(args):
    with open(CONFIG_YAML, "r") as f:
        cfg = yaml.safe_load(f)

    goat_path = args.goat_bench or DEFAULT_GOAT_BENCH
    split = args.split or DEFAULT_SPLIT
    content_dir = os.path.join(goat_path, split, "content")
    
    if not os.path.isdir(content_dir):
        print(f"ERROR: content directory not found: {content_dir}", file=sys.stderr)
        sys.exit(2)

    scene_files = sorted(fn for fn in os.listdir(content_dir) if fn.endswith(".json.gz"))
    scene_names = [fn.split(".")[0] for fn in scene_files]

    start_index = args.start_index or 0
    scene_names = scene_names[start_index:]

    out_root = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for each task
    tasks = []
    for scene_name in scene_names:
        tasks.append((
            scene_name, 
            goat_path, 
            split, 
            str(out_root), 
            args.top_k or DEFAULT_TOP_K, 
            args.save_size
        ))

    # --- PARALLEL CONFIGURATION ---
    # Adjust max_workers based on your GPU VRAM. 
    # Habitat scenes can take 1-2GB VRAM each. 
    # If you have a 24GB card, 4-6 workers is usually safe.
    MAX_WORKERS = 8

    print(f"[driver] Starting pool with {MAX_WORKERS} workers...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(spawn_worker_process, tasks)

    print("[driver] finished all scenes.")



# Worker: heavy lifting per scene
def run_worker(args):
    import numpy as np
    import habitat_sim
    import quaternion
    from PIL import Image
    from habitat_sim.utils import common as utils
    from tqdm import tqdm

    random.seed(0)
    np.random.seed(0)

    out_root = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"generate_{args.scene}.log"

    logger = logging.getLogger(f"dataset_gen.{args.scene}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f"Worker starting for scene: {args.scene}")

    with open(CONFIG_YAML, "r") as f:
        config = yaml.safe_load(f)

    goat_path = args.goat_bench or DEFAULT_GOAT_BENCH
    split = args.split or DEFAULT_SPLIT
    if "val" in split:
        hm3d_split = "val"
    else:
        hm3d_split = split

    def quat_from_episode(rotation_list):
        return utils.quat_from_coeffs(rotation_list).normalized()

    def offset_rotation_world_yaw(quat, degrees):
        rad = np.deg2rad(degrees)
        return (utils.quat_from_angle_axis(rad, np.array([0.0, 1.0, 0.0])) * quat).normalized()

    def forward_right_vectors(rot):
        fwd = quaternion.rotate_vectors(rot, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        right = quaternion.rotate_vectors(rot, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        return fwd, right

    def too_flat_depth(depth, flat_thresh=0.3, uniform_ratio=0.5):
        valid = depth[depth > 0]
        if valid.size == 0:
            return True
        mean = valid.mean()
        std = valid.std()
        if std < flat_thresh:
            return True
        near_mean = np.abs(valid - mean) < flat_thresh
        return (near_mean.sum() / float(valid.size)) > uniform_ratio

    def image_stats_from_np(rgb_np):
        gray = np.mean(rgb_np[..., :3], axis=2)
        return float(gray.mean()), float(gray.std())

    def linear_to_srgb(img_np):
        arr = img_np.astype(np.float32) 
        arr /= 255.0
        np.clip(arr, 0, 1, out=arr)
        np.power(arr, 1.0 / 2.2, out=arr)
        arr *= 255.0
        return arr.astype(np.uint8)

    visibility_cache = {}

    def pose_key(pos, rot):
        p = tuple(np.round(np.asarray(pos, dtype=np.float32), 3))
        r = (float(rot.x), float(rot.y), float(rot.z), float(rot.w))
        r = tuple(np.round(np.asarray(r, dtype=np.float32), 4))
        return p + r

    def get_obs(sim, agent, pos, rot):
        st = habitat_sim.AgentState(position=pos, rotation=rot)
        agent.set_state(st)
        obs = sim.get_sensor_observations()
        color = np.ascontiguousarray(obs["color_sensor"][..., :3])
        depth = np.ascontiguousarray(obs["depth_sensor"])
        sem = np.ascontiguousarray(obs["semantic_sensor"].astype(np.int32))
        return color, depth, sem

    def semantic_visible(sem_img, target_ids_array, min_pixels=50):
        if sem_img.size == 0 or target_ids_array.size == 0:
            return False
        
        mask = np.isin(sem_img, target_ids_array)
        count = np.count_nonzero(mask)
        return count >= min_pixels

    def get_semantic_pixel_count(sem_img, target_ids_array):
        if sem_img.size == 0 or target_ids_array.size == 0:
            return 0
        mask = np.isin(sem_img, target_ids_array)
        return np.count_nonzero(mask)

    class SemanticIndex:
        def __init__(self, sim):
            self.instance_ids_by_name = {}
            self.category_ids_by_name = {}
            self._build(sim)

        @staticmethod
        def _try_call(v):
            try:
                return v() if callable(v) else v
            except Exception:
                return None

        def _build(self, sim):
            scene = getattr(sim, "semantic_scene", None)
            if scene is None:
                return

            for i, obj in enumerate(scene.objects):
                if obj.category is None:
                    continue

                cname = str(obj.category.name()).lower()
                
                inst_id = None
                for attr in ("semantic_id", "semanticID"):
                    if hasattr(obj, attr):
                        inst_id = self._try_call(getattr(obj, attr))
                        if inst_id is not None:
                            break

                cat_id = None
                if hasattr(obj.category, "index"):
                    cat_id = self._try_call(getattr(obj.category, "index"))
                if inst_id is not None:
                    self.instance_ids_by_name.setdefault(cname, set()).add(int(inst_id))
                if cat_id is not None:
                    self.category_ids_by_name.setdefault(cname, set()).add(int(cat_id))

        def ids_for_query(self, qname):
            q = qname.lower().split("_")[0]
            inst = set()

            for k in self.instance_ids_by_name:
                if q in k:
                    inst |= self.instance_ids_by_name[k]

            return np.array(list(inst), dtype=np.int32)

    # Navmesh helpers
    def _iter_navmesh_candidates(root_dir):
        exts = (".navmesh", ".navmesh.bin", ".basis.navmesh", ".basis.navmesh.bin")
        for name in os.listdir(root_dir):
            p = os.path.join(root_dir, name)
            if os.path.isfile(p) and any(name.endswith(e) for e in exts):
                yield p
        parent = os.path.dirname(root_dir)
        if os.path.isdir(parent):
            for name in os.listdir(parent):
                p = os.path.join(parent, name)
                if os.path.isfile(p) and any(name.endswith(e) for e in exts):
                    yield p
        for dp, dn, files in os.walk(root_dir):
            for name in files:
                if any(name.endswith(e) for e in exts):
                    yield os.path.join(dp, name)

    def ensure_navmesh(sim, scene_dir):
        for cand in _iter_navmesh_candidates(scene_dir):
            try:
                if sim.pathfinder.load_nav_mesh(cand):
                    logger.info(f"Loaded NavMesh: {cand}")
                    return True
            except Exception:
                pass
        try:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius = 0.2
            settings.agent_height = AGENT_HEIGHT
            settings.agent_max_climb = 0.2
            settings.agent_max_slope = 45.0
            ok = sim.recompute_navmesh(sim.pathfinder, settings)
            if ok and sim.pathfinder.is_loaded:
                logger.info("Recomputed NavMesh in-memory.")
                return True
        except Exception as e:
            logger.warning(f"NavMesh recompute failed: {e}")
        return False

    def try_step_from(sim, start, end):
        v = sim.pathfinder.try_step(start, end)
        try:
            arr = np.array([v[0], v[1], v[2]], dtype=np.float32)
        except Exception:
            return None
        if not np.isfinite(arr).all():
            return None
        return arr

    def evaluate_pose(sim, agent, pos, rot, target_ids_array,
                      min_brightness, min_texture, min_sem_pixels):
        key = pose_key(pos, rot)
        if key in visibility_cache:
            return visibility_cache[key]

        rgb_np, depth, sem = get_obs(sim, agent, pos, rot)
        mean, std = image_stats_from_np(rgb_np)

        if mean < min_brightness or std < min_texture or too_flat_depth(depth):
            visible = False
        else:
            visible = semantic_visible(sem, target_ids_array, min_pixels=min_sem_pixels)

        visibility_cache[key] = (visible, mean, std, rgb_np, depth, sem)
        return visibility_cache[key]

    def find_related_frontier_semantic(
        sim,
        agent,
        pos,
        rot,
        obj_name,
        target_ids_array,
        min_sem_pixels_dynamic,
        obj_center=None,
        step_sizes=(0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
        min_brightness=12.0,
        min_texture=8.0,
    ):
        """
        Optimized quality-preserving frontier search with dynamic pixel threshold.
        """

        if obj_center is None:
            logger.warning(f"[{obj_name}] No object center provided — skipping related frontier.")
            return None, None, None

        if target_ids_array.size == 0:
            return None, None, None

        # Must be visible at starting pose
        rgb0, depth0, sem0 = get_obs(sim, agent, pos, rot)
        mean0, std0 = image_stats_from_np(rgb0)
        
        # Strict start check
        if not (mean0 >= min_brightness and std0 >= min_texture):
            return None, None, None
            
        if not semantic_visible(sem0, target_ids_array, min_pixels=min_sem_pixels_dynamic):
             # Even if main view was visible, if it doesn't meet the dynamic threshold, skip
             return None, None, None

        obj_center = np.array(obj_center, dtype=np.float32)
        start_pos = np.array(pos, dtype=np.float32)

        fwd, right = forward_right_vectors(rot)
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        best = None
        best_dist = -np.inf

        # Shuffle directions to avoid bias, though we check all
        random.shuffle(directions) 

        for dx, dz in directions:
            move_dir = (fwd * dz + right * dx).astype(np.float32)
            n = np.linalg.norm(move_dir)
            if n < 1e-8: continue
            move_dir /= n

            last_visible_pose = None
            last_rgb = None

            for step in step_sizes:
                new_pos = start_pos + move_dir * step
                stepped = try_step_from(sim, start_pos, new_pos)
                if stepped is None:
                    continue

                rgb_np, depth_np, sem_np = get_obs(sim, agent, stepped, rot)
                mean, std = image_stats_from_np(rgb_np)

                if mean < min_brightness or std < min_texture or too_flat_depth(depth_np):
                    continue

                # Use the dynamic threshold derived from Main View
                if semantic_visible(sem_np, target_ids_array, min_pixels=min_sem_pixels_dynamic):
                    last_visible_pose = (stepped, rot)
                    last_rgb = rgb_np

            if last_visible_pose:
                dist = np.linalg.norm(last_visible_pose[0] - obj_center)
                if dist > best_dist:
                    best_dist = dist
                    best = (*last_visible_pose, last_rgb)

        if best is None:
            return None, None, None
        return best


    def find_unrelated_frontier_semantic(
        sim, agent, rel_pos, rel_rot, obj_name, target_ids_array,
        yaw_jitter=(-10, 0, 10),
        min_brightness=12.0, min_texture=6.0,
    ):
        if rel_pos is None or rel_rot is None or target_ids_array.size == 0:
            return None, None, None
        
        for base in [90, 135, 180, -135, -90]:
            base_rot = offset_rotation_world_yaw(rel_rot, base)
            for jitter in yaw_jitter:
                rot_try = offset_rotation_world_yaw(base_rot, jitter)
                
                key = pose_key(rel_pos, rot_try)
                if key in visibility_cache:
                    visible, mean, std, rgb_np, depth, sem = visibility_cache[key]
                else:
                    rgb_np, depth, sem = get_obs(sim, agent, rel_pos, rot_try)
                    mean, std = image_stats_from_np(rgb_np)
                    visibility_cache[key] = (False, mean, std, rgb_np, depth, sem)

                if mean >= min_brightness and std >= min_texture and not too_flat_depth(depth):
                    count = get_semantic_pixel_count(sem, target_ids_array)
                    if count < 50:
                        return rel_pos, rot_try, rgb_np

        return None, None, None

    try:
        content_path = os.path.join(goat_path, split, "content", f"{args.scene}.json.gz")
        if not os.path.exists(content_path):
            logger.error(f"Episodes file not found: {content_path}")
            return 2
        with gzip.open(content_path, "rt", encoding="utf-8") as f:
            episodes = json.load(f)
    except Exception as e:
        logger.exception(f"Failed to load episodes for {args.scene}: {e}")
        return 1

    goals = episodes.get("goals", {})

    try:
        scene_dir_candidates = [
            d for d in os.listdir(os.path.join(config["scene_data_path"], hm3d_split))
            if args.scene in d
        ]
        if not scene_dir_candidates:
            logger.warning(f"No local scene dir found for {args.scene}, skipping.")
            return 0
        scene_id = scene_dir_candidates[0]
        scene_mesh_path = os.path.join(
            config["scene_data_path"], hm3d_split, scene_id, f"{args.scene}.basis.glb"
        )
        if not os.path.exists(scene_mesh_path):
            logger.warning(f"Scene mesh not found: {scene_mesh_path}, skipping.")
            return 0

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_mesh_path
        sim_cfg.scene_dataset_config_file = config["scene_dataset_config_path"]

        if hasattr(sim_cfg, "gpu_device_id"):
            sim_cfg.gpu_device_id = 0

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        rgb_spec.resolution = [int(config["img_height"]), int(config["img_width"])]
        rgb_spec.position = [0.0, float(config["camera_height"]), 0.0]
        rgb_spec.orientation = [float(np.deg2rad(config["camera_tilt_deg"])), 0.0, 0.0]
        rgb_spec.hfov = float(config["hfov"])
        
        rgb_spec.gpu2gpu_transfer = False 

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        depth_spec.resolution = rgb_spec.resolution
        depth_spec.position = rgb_spec.position
        depth_spec.orientation = rgb_spec.orientation

        sem_spec = habitat_sim.CameraSensorSpec()
        sem_spec.uuid = "semantic_sensor"
        sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sem_spec.resolution = rgb_spec.resolution
        sem_spec.position = rgb_spec.position
        sem_spec.orientation = rgb_spec.orientation

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec, sem_spec]

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        logger.info(f"Initializing simulator for scene {args.scene}")
        sim = habitat_sim.Simulator(cfg)

    except Exception as e:
        logger.exception(f"Failed to initialize simulator for {args.scene}: {e}")
        return 1

    try:
        if not hasattr(sim, "semantic_scene") or sim.semantic_scene is None or len(sim.semantic_scene.objects) == 0:
            logger.info(f"Scene {args.scene} has no semantic annotations — skipping.")
            sim.close()
            return 0

        scene_dir = os.path.dirname(scene_mesh_path)
        if not ensure_navmesh(sim, scene_dir):
            logger.warning(f"NavMesh unavailable for {args.scene}, skipping.")
            sim.close()
            return 0

        agent = sim.initialize_agent(0)
        sem_index = SemanticIndex(sim)

        rgb_root = out_root / "rgb" / args.scene
        rgb_root.mkdir(parents=True, exist_ok=True)

        for obj_id, goal_list in goals.items():
            for goal_instance in goal_list:
                obj_name = goal_instance.get("object_id", goal_instance.get("object_category", str(obj_id)))
                cat_name = goal_instance.get("object_category", str(obj_id))
                viewpoints = goal_instance.get("view_points", [])
                if not viewpoints:
                    continue

                target_ids_array = sem_index.ids_for_query(obj_name)
                if target_ids_array.size == 0:
                    logger.warning(f"Skipping object '{obj_name}' - No matching semantic IDs found in scene.")
                    continue

                enum_vps = [(i, vp) for i, vp in enumerate(viewpoints)]
                enum_vps.sort(key=lambda t: t[1].get("iou", 1e9), reverse=True)
                selected_vps = enum_vps[:(args.top_k or DEFAULT_TOP_K)]

                obj_dir = rgb_root / obj_name
                obj_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"{args.scene}:{obj_name} -> {len(selected_vps)} high IoU viewpoints")

                for vp_idx, vp in tqdm(selected_vps, desc=f"{args.scene}:{obj_name}", ncols=80):
                    pos = np.array(vp["agent_state"]["position"], dtype=np.float32)
                    rot = quat_from_episode(vp["agent_state"]["rotation"])
                    obj_center = goal_instance.get("object_center") or goal_instance.get("position")

                    # Capture Main View
                    main_rgb_np, main_depth, main_sem = get_obs(sim, agent, pos, rot)
                    
                    # Dynamic Pixel Calculation from Main View
                    main_pixel_count = get_semantic_pixel_count(main_sem, target_ids_array)

                    if main_pixel_count == 0:
                         logger.info(f"Object {obj_name} not visible in main view {vp_idx}")
                         continue
                    
                    dynamic_min_pixels = min(300, max(20, int(main_pixel_count * 0.2)))

                    # Related frontier search
                    rpos, rrot, r_rgb_np = find_related_frontier_semantic(
                        sim, agent, pos, rot,
                        obj_name.split("_")[0],
                        target_ids_array,
                        dynamic_min_pixels,
                        obj_center=obj_center,
                    )
                    if r_rgb_np is None:
                        continue

                    # Unrelated frontier search
                    upos, urot, u_rgb_np = find_unrelated_frontier_semantic(
                        sim, agent, rpos, rrot,
                        obj_name.split("_")[0],
                        target_ids_array,
                    )
                    if u_rgb_np is None:
                        continue

                    def np_to_pil(rgb_np):
                        rgb_gamma = linear_to_srgb(rgb_np)
                        img = Image.fromarray(rgb_gamma)
                        img = img.resize(tuple(args.save_size), Image.LANCZOS)
                        return img

                    main_img = np_to_pil(main_rgb_np)
                    r_img = np_to_pil(r_rgb_np)
                    u_img = np_to_pil(u_rgb_np)

                    main_img.save(obj_dir / f"{vp_idx:04d}_main.png", format="PNG", optimize=False, compress_level=1)
                    r_img.save(obj_dir / f"{vp_idx:04d}_frontier_related.png", format="PNG", optimize=False, compress_level=1)
                    u_img.save(obj_dir / f"{vp_idx:04d}_frontier_unrelated.png", format="PNG", optimize=False, compress_level=1)

                    del main_rgb_np, r_rgb_np, u_rgb_np, main_img, r_img, u_img
                    visibility_cache.clear()
                    gc.collect()

        sim.close()
        del sim, agent, sem_index
        gc.collect()
        logger.info(f"Worker finished scene {args.scene}")
        return 0

    except Exception as e:
        logger.exception(f"Worker error for scene {args.scene}: {e}")
        try:
            sim.close()
        except Exception:
            pass
        gc.collect()
        return 1


def make_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--driver", action="store_true", help="Run driver that spawns per-scene workers")
    p.add_argument("--scene", type=str, help="Run single scene (worker mode)")
    p.add_argument("--goat-bench", type=str, default=None, help="GOAT bench path")
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--save-size", type=int, nargs=2, default=list(DEFAULT_SAVE_SIZE))
    p.add_argument("--start-index", type=int, default=0)
    return p


def main_entry():
    args = make_arg_parser().parse_args()
    if args.driver:
        run_driver(args)
    elif args.scene:
        rc = run_worker(args)
        sys.exit(rc)
    else:
        print("Specify --driver or --scene <name>", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main_entry()

