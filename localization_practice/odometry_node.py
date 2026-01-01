#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
import threading

class OdometryLocalizationNode(Node):
    def __init__(self):
        super().__init__('odometry_localization_node')
        
        # Robot parameters
        self.wheel_base = 0.5  # Distance between wheels (meters)
        
        # State: [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])
        
        # Covariance matrix (3x3) - initially zero since we know exact position
        self.covariance = np.zeros((3, 3))
        
        # Store trajectory for plotting
        self.trajectory = [self.state.copy()]
        self.covariances = [self.covariance.copy()]
        
        # Sensor noise parameters (you can adjust these)
        self.k_r = 0.01  # Right wheel noise coefficient
        self.k_l = 0.01  # Left wheel noise coefficient
        
        self.get_logger().info('Odometry Localization Node started!')
        self.get_logger().info('Initial position: (0.0, 0.0), heading: 0.0 rad')
        self.get_logger().info('='*60)
    
    def motion_model(self, delta_sr, delta_sl):
        """
        Differential drive motion model
        Updates state and covariance based on wheel movements
        """
        # Extract current state
        x, y, theta = self.state
        
        # Calculate motion
        delta_s = (delta_sr + delta_sl) / 2.0  # Average distance
        delta_theta = (delta_sr - delta_sl) / self.wheel_base  # Change in orientation
        
        # Update state (using simplified motion model)
        if abs(delta_theta) < 1e-6:  # Straight line motion
            x_new = x + delta_s * np.cos(theta)
            y_new = y + delta_s * np.sin(theta)
            theta_new = theta
        else:  # Curved motion
            # Calculate radius of curvature
            R = delta_s / delta_theta
            x_new = x + R * (np.sin(theta + delta_theta) - np.sin(theta))
            y_new = y - R * (np.cos(theta + delta_theta) - np.cos(theta))
            theta_new = theta + delta_theta
        
        # Normalize angle to [-pi, pi]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        # Jacobian of motion model with respect to state
        G = np.array([
            [1, 0, -delta_s * np.sin(theta)],
            [0, 1,  delta_s * np.cos(theta)],
            [0, 0, 1]
        ])
        
        # Jacobian of motion model with respect to controls (wheel movements)
        # This represents how wheel movements affect the state
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        V = np.array([
            [0.5 * cos_theta, 0.5 * cos_theta],
            [0.5 * sin_theta, 0.5 * sin_theta],
            [1.0 / self.wheel_base, -1.0 / self.wheel_base]
        ])
        
        # Sensor noise covariance (diagonal matrix)
        # Noise proportional to distance traveled
        sigma_r = self.k_r * abs(delta_sr)
        sigma_l = self.k_l * abs(delta_sl)
        M = np.diag([sigma_r**2, sigma_l**2])
        
        # Update covariance using error propagation
        # Σ_new = G * Σ_old * G^T + V * M * V^T
        self.covariance = G @ self.covariance @ G.T + V @ M @ V.T
        
        # Update state
        self.state = np.array([x_new, y_new, theta_new])
        
        # Store for trajectory
        self.trajectory.append(self.state.copy())
        self.covariances.append(self.covariance.copy())
        
        return x_new, y_new, theta_new
    
    def get_user_input(self):
        """
        Get wheel distances from user
        """
        print("\n" + "="*60)
        print("Enter wheel distances (or 'exit' to quit)")
        
        try:
            user_input = input("Right wheel distance (Δsr) in meters (or 'exit'): ").strip().lower()
            if user_input == 'exit':
                return None, None, True
            
            delta_sr = float(user_input)
            
            user_input = input("Left wheel distance (Δsl) in meters: ").strip().lower()
            if user_input == 'exit':
                return None, None, True
            
            delta_sl = float(user_input)
            
            return delta_sr, delta_sl, False
            
        except ValueError:
            self.get_logger().error('Invalid input! Please enter numeric values.')
            return None, None, False
        except KeyboardInterrupt:
            return None, None, True
    
    def plot_trajectory(self):
        """
        Plot the robot's trajectory with error ellipses
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract x, y coordinates
        x_coords = [state[0] for state in self.trajectory]
        y_coords = [state[1] for state in self.trajectory]
        
        # Plot trajectory
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Robot Trajectory')
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=15, label='Start Position')
        ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=15, label='End Position')
        
        # Plot error ellipses at each position
        for i, (state, cov) in enumerate(zip(self.trajectory, self.covariances)):
            # Extract position covariance (2x2)
            pos_cov = cov[0:2, 0:2]
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(pos_cov)
            
            # Calculate ellipse parameters (95% confidence - 2.447 sigma)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width = 2 * np.sqrt(5.991 * eigenvalues[0])  # Chi-square 95% for 2 DOF
            height = 2 * np.sqrt(5.991 * eigenvalues[1])
            
            # Create and add ellipse
            ellipse = Ellipse(xy=(state[0], state[1]), 
                            width=width, 
                            height=height, 
                            angle=angle,
                            facecolor='none',
                            edgecolor='red',
                            linewidth=1,
                            alpha=0.5)
            ax.add_patch(ellipse)
            
            # Add step numbers
            if i % max(1, len(self.trajectory) // 10) == 0 or i == len(self.trajectory) - 1:
                ax.text(state[0], state[1], f'{i}', fontsize=8, ha='center')
        
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        ax.set_title('Robot Trajectory with 95% Confidence Error Ellipses', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig('robot_trajectory.png', dpi=300, bbox_inches='tight')
        self.get_logger().info('Trajectory plot saved as "robot_trajectory.png"')
        plt.show()
    
    def print_statistics(self):
        """
        Print final statistics
        """
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Total steps: {len(self.trajectory) - 1}")
        print(f"Final position: x={self.state[0]:.4f} m, y={self.state[1]:.4f} m")
        print(f"Final heading: {np.degrees(self.state[2]):.2f} degrees")
        print(f"\nFinal position uncertainty (standard deviation):")
        print(f"  σ_x = {np.sqrt(self.covariance[0, 0]):.4f} m")
        print(f"  σ_y = {np.sqrt(self.covariance[1, 1]):.4f} m")
        print(f"  σ_θ = {np.degrees(np.sqrt(self.covariance[2, 2])):.4f} degrees")
        print("="*60)
    
    def run(self):
        """
        Main loop to get user input and update odometry
        """
        step = 0
        
        while rclpy.ok():
            delta_sr, delta_sl, should_exit = self.get_user_input()
            
            if should_exit:
                self.get_logger().info('Exiting and generating plots...')
                break
            
            if delta_sr is not None and delta_sl is not None:
                # Update odometry
                x, y, theta = self.motion_model(delta_sr, delta_sl)
                step += 1
                
                # Log the update
                self.get_logger().info(f'Step {step}: Position = ({x:.4f}, {y:.4f}), '
                                      f'Heading = {np.degrees(theta):.2f}°')
                self.get_logger().info(f'  Uncertainty: σ_x={np.sqrt(self.covariance[0,0]):.4f}, '
                                      f'σ_y={np.sqrt(self.covariance[1,1]):.4f}')
        
        # Print statistics and plot
        self.print_statistics()
        self.plot_trajectory()


def main(args=None):
    rclpy.init(args=args)
    
    node = OdometryLocalizationNode()
    
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
