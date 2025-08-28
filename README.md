## Go2 PPO: Train, Sim, and ROS2 Control

### Prerequisites
- Python environment with: genesis, rsl-rl-lib==2.2.4, torch, zenoh-python
- ROS 2 (sourced, e.g., `source /opt/ros/$ROS_DISTRO/setup.bash`)
- Optional for keyboard: `teleop_twist_keyboard`

Install keyboard teleop (Ubuntu):
```bash
sudo apt update && sudo apt install -y ros-$ROS_DISTRO-teleop-twist-keyboard
```

### 1) Train
This trains the Go2 walking policy and saves logs under `logs/<exp_name>`.
```bash
cd /home/shipengl/genesis_ros/
python -m genesis_ros.ppo.go2_train -e go2-walking -B 4096 --max_iterations 101
```

Notes:
- Change `-B` for number of parallel envs and `--max_iterations` as needed.
- Checkpoints are saved as `genesis_ros/logs/go2-walking/model_*.pt`. Config is saved in `cfgs.pkl`.

### 2) Run Simulation (Zenoh/Python or ROS2 mode) (run in genesis conda environments)
By default, the sim runs the trained policy and exposes topics via Zenoh. You can also select ROS2 interface.

Python/Zenoh interface (recommended for use with the bridge):
```bash
cd /home/shipengl/genesis_ros/
python -m genesis_ros.ppo.genesis_simulation_go2_zenoh -d gpu -e go2-walking -i python
```

ROS2 interface (direct control without using the ppo, it will be transferred using zenoh_ros2_bridge):
```bash
cd /home/shipengl/genesis_ros/genesis_ros/ppo
python -m genesis_ros.ppo.genesis_simulation_go2_zenoh -d gpu -e go2-walking -i ros2
```

Flags:
- `-d` device: `cpu` or `gpu`
- `-e` experiment name used in training
- `--ckpt` checkpoint index to load
- `-i` interface: `python` (Zenoh) or `ros2`

### 3) Start ROS2 ↔ Zenoh Bridge (only when sim is `-i python`)
This node bridges ROS2 `/cmd_vel` to Zenoh `cmd_vel`, and forwards sim observations to ROS2 topics like `/joint_states` and `/odom`.
```bash
# New terminal (ROS 2 sourced), has to be with the system python environments
cd /home/shipengl/genesis_ros/genesis_ros/ppo
python3 -m genesis_ros.ppo.zenoh_ros2_bridge
```

Published/Subscribed topics:
- ROS2 subscribe: `/cmd_vel` (geometry_msgs/Twist)
- ROS2 publish: `/joint_states` (sensor_msgs/JointState), `/odom` (nav_msgs/Odometry) (not tested)
- Zenoh forward: `cmd_vel` → sim, `control/observation` → ROS2 (not working now, need to broadcast the state messages. )

### 4) Control the Robot

Option A — Keyboard teleop (publishes to `/cmd_vel`):
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
Tips:
- Use `i`,`j`,`l`,`k`,`,` keys as indicated. Ensure the bridge is running if sim is in `-i python`.

Option B — Directly publish speeds with ROS2:
```bash
# Example: forward 0.5 m/s at 10 Hz
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

### Typical Workflows

Keyboard control via ROS2 (recommended):
1. Train or ensure `logs/go2-walking/model_*.pt` exists
2. Start sim with Python/Zenoh interface:
   ```bash
   python genesis_simulation_go2_zenoh.py -d gpu -e go2-walking --ckpt 100 -i python
   ```
3. Start the ROS2↔Zenoh bridge:
   ```bash
   python zenoh_ros2_bridge.py
   ```
4. Run keyboard teleop:
   ```bash
   ros2 run teleop_twist_keyboard teleop_twist_keyboard
   ```

Direct ROS2 (no bridge):
1. Start sim with `-i ros2`
2. Publish `/cmd_vel` via teleop or `ros2 topic pub`

### Troubleshooting
- If you see no response to `/cmd_vel`, confirm:
  - Sim is running and `-i python` when using the bridge
  - Bridge is running and ROS 2 is sourced
  - Topics exist: `ros2 topic list` should include `/cmd_vel`
- For GPU issues, try `-d cpu` to validate setup.
- Ensure `logs/<exp>/cfgs.pkl` and `model_*.pt` exist for the chosen `--ckpt`.


