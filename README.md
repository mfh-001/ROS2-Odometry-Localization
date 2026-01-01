# ROS 2 Odometry & Localization with Error Propagation

This repository implements a ROS 2 node for robot localization using a Differential Drive kinematics model. The system tracks the robot's pose $(x, y, \theta)$ and calculates the growth of spatial uncertainty (covariance) as the robot moves, visualizing the results with 95% confidence error ellipses.

## Overview

The project features a single high-performance ROS 2 node:
- **`odometry_node`**: A Python-based node that accepts user-defined wheel displacements ($\Delta s_r, \Delta s_l$), updates the robot's state using dead reckoning, and propagates sensor noise through a motion model.

## Engineering Logic: Odometry & Uncertainty

### 1. Kinematic Update
The robot's state is updated based on the average displacement ($\Delta s$) and the change in orientation ($\Delta \theta$):

$$\Delta s = \frac{\Delta s_r + \Delta s_l}{2}$$
$$\Delta \theta = \frac{\Delta s_r - \Delta s_l}{b}$$ 

(where $b$ is the wheel base)

For curved motion, the node uses the Exact Integration model to determine the new position, ensuring high accuracy over long trajectories.

### 2. Error Propagation
To track uncertainty, the node implements a **Linearized Error Propagation** model. The covariance matrix $\Sigma$ is updated at every step using:

$$\Sigma_{new} = G \Sigma_{old} G^T + V M V^T$$

- **G**: Jacobian of the motion model with respect to the state.
- **V**: Jacobian of the motion model with respect to the control inputs.
- **M**: Noise covariance (proportional to the distance traveled).



## Tech Stack

- **Middleware:** ROS 2 (Humble/Foxy)
- **Language:** Python 3 (NumPy, Matplotlib)
- **Tools:** Colcon, ament_python, Linux (Ubuntu)

## 📂 Project Structure

- **localization_practice/odometry_node.py**: The main logic handling kinematics and covariance math.
- **robot_trajectory.png**: Visual output generated upon exit showing the path and error ellipses.
- **package.xml & setup.py**: ROS 2 Python package configuration.

## Visual Result

Upon exiting the node, a plot is generated showing the robot's path. The **Red Ellipses** represent the 95% confidence interval of the robot's position at that specific step.

<img width="1045" height="508" alt="Screenshot 2026-01-01 at 4 26 09 PM" src="https://github.com/user-attachments/assets/712ef91c-7eb8-4fd0-be14-0b639ff085f3" />


---
## ⚠️ Academic Integrity
This repository is intended solely as a piece to showcase my learning journey. 
If you are a student working on a similar assignment: **do not copy this code.** Plagiarism is a serious offense that can lead to expulsion. Use this only as a conceptual reference to understand odometry & localization with error propagation in ROS 2.
