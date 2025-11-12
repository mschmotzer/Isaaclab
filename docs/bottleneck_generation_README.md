# Bottleneck-Focused Data Generation for Isaac Lab

This document explains how to use the bottleneck-focused data generation system that integrates your spherical sampling logic with Isaac Lab's MIMIC framework.

## Overview

The bottleneck data generation system focuses on critical states in demonstrations to generate more challenging and diverse training data. It uses spherical coordinate sampling around identified bottleneck states to create variations that help improve policy robustness.

## Files Created

1. **`bottleneck_data_generator.py`** - Core bottleneck data generator class
2. **`generate_dataset_bottleneck.py`** - Modified generation script with bottleneck support
3. **`prepare_bottleneck_data.py`** - Utility to prepare HDF5 files with critical state annotations
4. **`run_bottleneck_generation.sh`** - Example usage script

## Workflow

### Step 1: Prepare Your Demonstration Data

First, you need HDF5 files with critical state annotations. Use the preparation script:

```bash
# Annotate your existing demonstration files with critical states
python scripts/imitation_learning/isaaclab_mimic/prepare_bottleneck_data.py \
    --input_dir ./datasets/original_demos \
    --output_dir ./datasets/bottleneck_demos \
    --annotation_method combined \
    --pattern "*.hdf5"
```

This will create annotated HDF5 files with an `augment_states` field marking critical states.

**Annotation Methods:**
- `reward_threshold` - Mark high-reward states as critical
- `action_variance` - Mark states with high action variance
- `gripper_changes` - Mark states around gripper action changes
- `pose_changes` - Mark states with rapid end-effector movements
- `combined` - Combine multiple methods (recommended)

### Step 2: Run Bottleneck Data Generation

Use the bottleneck-focused generation script:

```bash
python scripts/imitation_learning/isaaclab_mimic/generate_dataset_bottleneck.py \
    --task Isaac-Stack-Cube-Franka-IK-Abs-Mimic-v0 \
    --input_file ./datasets/source_demonstrations.hdf5 \
    --output_file ./datasets/bottleneck_augmented.hdf5 \
    --num_envs 4 \
    --generation_num_trials 100 \
    --use_bottleneck_mode \
    --bottleneck_data_dir ./datasets/bottleneck_demos \
    --bottleneck_pattern "*.hdf5" \
    --headless
```

**Key Arguments:**
- `--use_bottleneck_mode` - Enable bottleneck-focused generation
- `--bottleneck_data_dir` - Directory with annotated HDF5 files
- `--bottleneck_pattern` - File pattern for bottleneck files (default: `*.hdf5`)

### Step 3: Compare Results

You can compare bottleneck-focused vs. regular generation:

```bash
# Run the example script
./scripts/imitation_learning/isaaclab_mimic/run_bottleneck_generation.sh
```

## How It Works

### Bottleneck Selection

1. **Load Critical States**: The system loads HDF5 files and identifies states marked with `augment_states = 1`
2. **Probability Sampling**: States are sampled with inverse probability to their usage count (less-used states have higher probability)
3. **Spherical Variations**: Around selected critical states, spherical coordinate sampling generates pose variations

### Spherical Sampling

Your original spherical sampling logic is preserved:

```python
# Sample spherical coordinates
r = np.random.uniform(0.2, 0.5)           # Distance
theta = np.deg2rad(np.random.uniform(5, 80))  # Elevation angle
phi = np.random.uniform(0, 2 * np.pi)     # Azimuth angle

# Convert to Cartesian offsets
dx = r * np.sin(theta) * np.cos(phi)
dy = r * np.sin(theta) * np.sin(phi)
dz = r * np.cos(theta)

# Sample rotational variations
dr = np.random.uniform(-np.deg2rad(180), np.deg2rad(180))
dp = np.random.uniform(-np.deg2rad(180), np.deg2rad(180))
dyaw = np.random.uniform(-np.deg2rad(180), np.deg2rad(180))
```

### Integration with Isaac Lab

The bottleneck system integrates seamlessly with Isaac Lab's existing infrastructure:

- **Async Generation**: Multiple bottleneck generators run in parallel
- **Waypoint System**: Spherical variations are converted to Isaac Lab waypoints
- **Environment Coordination**: Works with multi-arm setups and subtask constraints
- **Recording**: Generated data is automatically recorded and exported

## Expected HDF5 Structure

Your bottleneck HDF5 files should follow this structure:

```
demo_0/
├── actions                    # Robot actions [T, action_dim]
├── observations/
│   ├── robot_state/
│   │   ├── ee_pos            # End-effector position [T, 3]
│   │   ├── ee_quat           # End-effector quaternion [T, 4]
│   │   └── gripper_width     # Gripper width [T, 1]
│   └── parts_poses           # Object poses [T, N_objects, 7]
├── rewards                   # Rewards [T]
├── success                   # Success flag [1]
└── augment_states           # Critical state annotations [T] (0 or 1)
demo_1/
├── ...
```

## Customization

### Custom Critical State Detection

You can modify `prepare_bottleneck_data.py` to implement your own critical state detection:

```python
def your_custom_annotation_method(actions, ee_poses, rewards):
    # Your custom logic here
    augment_states = np.zeros(len(actions), dtype=int)
    # Mark specific timesteps as critical
    return augment_states
```

### Custom Spherical Sampling

Modify the `generate_spherical_variations` method in `BottleneckDataGenerator`:

```python
def generate_spherical_variations(self, start_ee_pos, start_ee_quat, num_steps=10):
    # Your custom spherical sampling logic
    # Modify ranges, distributions, etc.
    pass
```

## Troubleshooting

1. **No Critical States Found**: If no critical states are detected, the system falls back to regular generation
2. **Missing HDF5 Fields**: The system handles missing fields gracefully with warnings
3. **Type Errors**: Ensure your HDF5 files have the correct data types (float32/float64 for poses, int for augment_states)

## Performance

- **Parallel Generation**: Uses multiple async workers for faster data generation
- **Memory Efficient**: Loads HDF5 data on-demand rather than keeping everything in memory
- **Scalable**: Can handle large numbers of bottleneck files and demonstrations

## Integration with Your Existing Workflow

This system preserves your existing Isaac Lab workflow while adding bottleneck functionality:

1. **Regular Mode**: Use without `--use_bottleneck_mode` for standard generation
2. **Bottleneck Mode**: Add bottleneck flags to focus on critical states
3. **Hybrid Mode**: Generate both regular and bottleneck data for comparison

The bottleneck system is designed to be a drop-in enhancement to your existing Isaac Lab MIMIC setup.
