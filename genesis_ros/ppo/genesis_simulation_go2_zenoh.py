import argparse
from genesis_ros.ppo.go2_env import Go2Env
import genesis as gs  # type: ignore
import torch
from genesis_ros.ros2_interface import builtin_interfaces, rosgraph_msgs, torch_msgs
from genesis_ros.ros2_interface import ROS2Interface
from genesis_ros.ros2_interface import torch_msgs
from genesis_ros.ros2_interface.topic_interfaces import TopicInterface, NopInterface
import pickle
import shutil
import os
from rsl_rl.runners import OnPolicyRunner
from typing import Union
import zenoh
import json


def eval(
    exp_name: str,
    ckpt: int,
    show_viewer: bool = True,
    urdf_path: str = "urdf/go2/urdf/go2.urdf",
    device: str = "gpu",
    interface: str = "python",
):
    if device == "cpu":
        gs.init(logging_level="warning", backend=gs.cpu)
    elif device == "gpu":
        gs.init(logging_level="warning", backend=gs.gpu)
    else:
        raise ValueError("Invalid device specified. Choose 'cpu' or 'gpu'.")

    topic_interface: TopicInterface
    if interface == "ros2":
        topic_interface = ROS2Interface(zenoh_config=zenoh.Config())
    if interface == "python":
        topic_interface = NopInterface()

    log_dir = f"logs/{exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"logs/{exp_name}/cfgs.pkl", "rb")
    )
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    step = 0
    
    # Initialize command state for NopInterface mode
    current_command = torch.zeros(3, device=gs.device)  # [lin_vel_x, lin_vel_y, ang_vel]
    command_received = False
    
    def on_twist_command(sample):
        """Callback for receiving Zenoh twist commands"""
        nonlocal current_command, command_received
        try:
            # Convert ZBytes to bytes and parse the twist command message
            payload_bytes = bytes(sample.payload)
            data = payload_bytes.decode('utf-8')
            twist_data = json.loads(data)
            
            # Extract linear and angular velocities
            linear = twist_data.get('linear', {})
            angular = twist_data.get('angular', {})
            
            # Update current command
            current_command[0] = float(linear.get('x', 0.0))  # lin_vel_x
            current_command[1] = float(linear.get('y', 0.0))  # lin_vel_y  
            current_command[2] = float(angular.get('z', 0.0))  # ang_vel
            
            # Clamp values to command ranges
            current_command[0] = torch.clamp(
                current_command[0], 
                env.command_cfg["lin_vel_x_range"][0], 
                env.command_cfg["lin_vel_x_range"][1]
            )
            current_command[1] = torch.clamp(
                current_command[1], 
                env.command_cfg["lin_vel_y_range"][0], 
                env.command_cfg["lin_vel_y_range"][1]
            )
            current_command[2] = torch.clamp(
                current_command[2], 
                env.command_cfg["ang_vel_range"][0], 
                env.command_cfg["ang_vel_range"][1]
            )
            
            command_received = True
            print(f"Received twist command: lin_vel_x={current_command[0]:.2f}, "
                  f"lin_vel_y={current_command[1]:.2f}, ang_vel={current_command[2]:.2f}")
                  
        except Exception as e:
            print(f"Error parsing twist command: {e}")
    
    # Set up Zenoh session and subscriber for NopInterface mode
    session = None
    if interface == "python":
        zenoh.init_log_from_env_or("info")
        config = zenoh.Config()
        session = zenoh.open(config)
        command_subscriber = session.declare_subscriber("cmd_vel", on_twist_command)
        print("Subscribing to twist commands on cmd_vel topic")
    
    with torch.no_grad():
        topic_interface.subscribe("control/action", torch_msgs.msg.FP32Tensor)
        while True:
            sec = step * env.dt
            topic_interface.publish(
                "clock",
                rosgraph_msgs.msg.Clock(
                    clock=builtin_interfaces.msg.Time(
                        sec=int(sec), nanosec=int((sec - int(sec)) * 1e9)
                    )
                ),
            )
            topic_interface.publish(
                "control/observation", torch_msgs.msg.from_torch_tensor(obs)
            )
            
            # Handle custom commands in NopInterface mode
            if interface == "python":
                if command_received:
                    # Set the commands in the environment
                    env.commands[0, :] = current_command
                    print(f"Using twist command: {current_command}")
                else:
                    # If no command received, move forward with default velocity
                    default_forward_vel = 0.0  # Default forward velocity in m/s
                    env.commands[0, :] = torch.tensor([default_forward_vel, 0.0, 0.0], device=gs.device)
                    print(f"Using default command: forward_vel={default_forward_vel}")
                
                actions = policy(obs)
                
            elif type(topic_interface) == ROS2Interface:
                topic_interface.spin()
                actions = topic_interface.spin_until_subscribe_new_data(
                    "control/action"
                ).to_torch_tensor()
            
            obs, rews, dones, infos = env.step(actions)
            if dones[0]:
                break
            step += 1
    
    # Clean up Zenoh session
    if session:
        session.close()
    
    gs.destroy()


def cli_entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        help="Specify device which you want to run PPO and simulation.",
        type=str,
        choices=["cpu", "gpu"],
        required=True,
    )
    parser.add_argument("-e", "--exp_name", type=str, default="genesis_ros_ppo")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument(
        "--urdf_path",
        help="Path to the URDF file",
        type=str,
        default="urdf/go2/urdf/go2.urdf",
    )
    parser.add_argument(
        "-i",
        "--interface",
        help="message passing interface",
        type=str,
        default="python",
        choices=["python", "ros2"],
    )
    args = parser.parse_args()
    eval(
        exp_name=args.exp_name,
        ckpt=args.ckpt,
        urdf_path=args.urdf_path,
        show_viewer=True,
        device=args.device,
        interface=args.interface,
    )


if __name__ == "__main__":
    cli_entrypoint()