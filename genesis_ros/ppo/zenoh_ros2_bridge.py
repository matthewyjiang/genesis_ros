#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import zenoh
import json
import torch
import numpy as np
from genesis_ros.ros2_interface import torch_msgs

class ZenohROS2Bridge(Node):
    def __init__(self):
        super().__init__('zenoh_ros2_bridge')
        
        # Initialize Zenoh
        zenoh.init_log_from_env_or("info")
        self.config = zenoh.Config()
        self.session = zenoh.open(self.config)
        
        # ROS2 publishers and subscribers
        self.twist_subscriber = self.create_subscription(
            Twist,
            'cmd_vel',
            self.on_ros2_twist,
            10
        )
        
        # Publish robot state to ROS2
        self.joint_state_publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )
        
        self.odometry_publisher = self.create_publisher(
            Odometry,
            'odom',
            10
        )
        
        # Zenoh subscribers and publishers
        self.zenoh_twist_publisher = self.session.declare_publisher("cmd_vel")
        self.zenoh_action_subscriber = self.session.declare_subscriber("control/action", self.on_zenoh_action)
        self.zenoh_observation_subscriber = self.session.declare_subscriber("control/observation", self.on_zenoh_observation)
        
        # Joint names for the robot (you may need to adjust these based on your robot)
        self.joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        
        self.get_logger().info("Zenoh-ROS2 Bridge started!")
        self.get_logger().info("Subscribing to /cmd_vel (ROS2) -> publishing to cmd_vel (Zenoh)")
        self.get_logger().info("Subscribing to control/observation (Zenoh) -> publishing to /joint_states and /odom (ROS2)")
        
    def on_ros2_twist(self, msg):
        """Convert ROS2 Twist message to Zenoh JSON and publish"""
        try:
            # Convert ROS2 Twist to JSON format
            twist_data = {
                "linear": {
                    "x": float(msg.linear.x),
                    "y": float(msg.linear.y),
                    "z": float(msg.linear.z)
                },
                "angular": {
                    "x": float(msg.angular.x),
                    "y": float(msg.angular.y),
                    "z": float(msg.angular.z)
                }
            }
            
            # Publish to Zenoh
            json_data = json.dumps(twist_data)
            self.zenoh_twist_publisher.put(json_data.encode('utf-8'))
            
            self.get_logger().debug(f"Published twist to Zenoh: {twist_data}")
            
        except Exception as e:
            self.get_logger().error(f"Error converting ROS2 twist to Zenoh: {e}")
    
    def on_zenoh_action(self, sample):
        """Handle Zenoh action messages (optional - for logging)"""
        try:
            # Convert ZBytes to bytes
            payload_bytes = bytes(sample.payload)
            
            # Parse action message using deserialize
            # Try different tensor types to find the correct one
            tensor_types = [
                torch_msgs.msg.FP32Tensor, 
                torch_msgs.msg.FP64Tensor,
                torch_msgs.msg.INT32Tensor,
                torch_msgs.msg.INT64Tensor,
                torch_msgs.msg.INT16Tensor,
                torch_msgs.msg.INT8Tensor,
                torch_msgs.msg.UINT8Tensor
            ]
            
            for tensor_type in tensor_types:
                try:
                    action_msg = tensor_type.deserialize(payload_bytes)
                    actions = action_msg.to_torch_tensor()
                    self.get_logger().info(f"Successfully deserialized action message as {tensor_type.__name__}: {actions.shape}")
                    break
                except Exception as e:
                    continue
            else:
                self.get_logger().warn(f"Could not deserialize action message. Payload size: {len(payload_bytes)} bytes")
                # Print first few bytes for debugging
                if len(payload_bytes) > 0:
                    self.get_logger().debug(f"First 20 bytes: {payload_bytes[:20]}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing Zenoh action: {e}")
    
    def on_zenoh_observation(self, sample):
        """Convert Zenoh observation message to ROS2 messages"""
        try:
            # Convert ZBytes to bytes
            payload_bytes = bytes(sample.payload)
            
            # Parse observation message using deserialize
            # Try different tensor types to find the correct one
            tensor_types = [
                torch_msgs.msg.FP32Tensor, 
                torch_msgs.msg.FP64Tensor,
                torch_msgs.msg.INT32Tensor,
                torch_msgs.msg.INT64Tensor,
                torch_msgs.msg.INT16Tensor,
                torch_msgs.msg.INT8Tensor,
                torch_msgs.msg.UINT8Tensor
            ]
            
            for tensor_type in tensor_types:
                try:
                    obs_msg = tensor_type.deserialize(payload_bytes)
                    observations = obs_msg.to_torch_tensor()
                    self.get_logger().info(f"Successfully deserialized observation message as {tensor_type.__name__}: {observations.shape}")
                    break
                except Exception as e:
                    continue
            else:
                self.get_logger().warn(f"Could not deserialize observation message. Payload size: {len(payload_bytes)} bytes")
                return
            
            # Extract state information from observations
            # The observation format is: [base_ang_vel(3), projected_gravity(3), commands(3), dof_pos_diff(12), dof_vel(12), actions(12)]
            # We need to extract the relevant parts for joint states and odometry
            
            # Extract joint positions (dof_pos_diff + default_dof_pos)
            dof_pos_diff = observations[9:21].cpu().numpy()  # 12 joint position differences
            # Note: We need the default_dof_pos to get actual joint positions
            # For now, we'll use the differences as approximate positions
            
            # Extract joint velocities
            dof_vel = observations[21:33].cpu().numpy()  # 12 joint velocities
            
            # Extract base angular velocity
            base_ang_vel = observations[0:3].cpu().numpy()
            
            # Extract projected gravity (can be used to estimate orientation)
            projected_gravity = observations[3:6].cpu().numpy()
            
            # Publish joint states
            self.publish_joint_states(dof_pos_diff, dof_vel)
            
            # Publish odometry (using available data)
            self.publish_odometry(base_ang_vel, projected_gravity)
            
        except Exception as e:
            self.get_logger().error(f"Error processing Zenoh observation: {e}")
    
    def publish_joint_states(self, dof_pos_diff, dof_vel):
        """Publish joint states to ROS2"""
        try:
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.name = self.joint_names
            
            # Use position differences as approximate positions - convert to Python floats
            joint_state_msg.position = [float(x) for x in dof_pos_diff.tolist()]
            joint_state_msg.velocity = [float(x) for x in dof_vel.tolist()]
            
            # Set efforts to zero (not available in observations)
            joint_state_msg.effort = [0.0] * len(self.joint_names)
            
            self.joint_state_publisher.publish(joint_state_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing joint states: {e}")
    
    def publish_odometry(self, base_ang_vel, projected_gravity):
        """Publish odometry to ROS2"""
        try:
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "odom"
            odom_msg.child_frame_id = "base_link"
            
            # Set position to zero (not available in observations)
            odom_msg.pose.pose.position.x = 0.0
            odom_msg.pose.pose.position.y = 0.0
            odom_msg.pose.pose.position.z = 0.0
            
            # Set orientation to identity (not directly available in observations)
            odom_msg.pose.pose.orientation.x = 0.0
            odom_msg.pose.pose.orientation.y = 0.0
            odom_msg.pose.pose.orientation.z = 0.0
            odom_msg.pose.pose.orientation.w = 1.0
            
            # Set linear velocity to zero (not available in observations)
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0
            
            # Set angular velocity from observations - convert to Python floats
            odom_msg.twist.twist.angular.x = float(base_ang_vel[0])
            odom_msg.twist.twist.angular.y = float(base_ang_vel[1])
            odom_msg.twist.twist.angular.z = float(base_ang_vel[2])
            
            self.odometry_publisher.publish(odom_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing odometry: {e}")
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        self.session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    bridge = ZenohROS2Bridge()
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
