import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def visualization(pose_file, save='/home/tako/mesrwi/animation.mp4'):
    seq = np.load(pose_file)
    
    # MotionBERT; Human3.6M
    bones = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=12., azim=90)

    # 애니메이션 업데이트 함수
    def update(frame):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        frame_data = seq[frame]
        
        for bone in bones:
            joint1, joint2 = bone
            ax.plot([-frame_data[joint1, 0], -frame_data[joint2, 0]],
                    [-frame_data[joint1, 2], -frame_data[joint2, 2]],
                    [-frame_data[joint1, 1], -frame_data[joint2, 1]], 'ro-')

    # 애니메이션 생성
    ani = FuncAnimation(fig, update, frames=seq.shape[0], interval=40)
    ani.save(save, fps=20)

if __name__ == '__main__':
    seq = np.load('/home/tako/mesrwi/Pose2Muscle/dataset/Subject0/5kg_ground/29/pose3d.npy')[:, :]

    # bones = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 14], [3, 14], [14, 16], [12, 16], [12, 8], [12, 9], [6, 7], [7, 8], [9, 10], [10, 11], [12, 15], [15, 13]] # MIADataset; 17 skeleton
    bones = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]] # MotionBERT; Human3.6M

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=12., azim=90)

    # 애니메이션 업데이트 함수
    def update(frame):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        frame_data = seq[frame]
        
        for bone in bones:
            joint1, joint2 = bone
            ax.plot([-frame_data[joint1, 0], -frame_data[joint2, 0]],
                    [-frame_data[joint1, 2], -frame_data[joint2, 2]],
                    [-frame_data[joint1, 1], -frame_data[joint2, 1]], 'ro-')

    # 애니메이션 생성
    ani = FuncAnimation(fig, update, frames=seq.shape[0], interval=40)
    ani.save('/home/tako/mesrwi/animation.mp4',fps=20)

    # 애니메이션 실행
    plt.show()