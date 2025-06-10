import time
import mujoco
from mujoco import viewer
import numpy as np
import os
import matplotlib.pyplot as plt
from mjcb_sensor.linux import TSensor

TSensor.register_sensor_callback()      # 注册回调

# 目标位置关节角度
target_qpos = np.array([0.514,0.686,0.539,  0.664,0.948,0.795,  0.748,0.997,0.847,  0.687,0.916,0.762,  0.416,0.924,0.77])
current_qpos = np.zeros((15,))

# 参数配置
step_ratio = 0.01
max_iterations = 1000
tolerance = 0.001          # 允许的位置误差

def set_ctrl(name, value):
    id = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    Data.ctrl[id] = value

# 绘制折线
def plt_fc():
    current_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(current_path, "./sensor_data/tactile.png")

    plt.figure(figsize=(12,8))
    data = np.array(Fc_data)
    x = [i for i in range(len(Fc_data))]

    finger_word = ("th", "ff", "mf", "rf", "lf")
    for i in range(5):
        plt.plot(x, data[:,i*2], label=f"{finger_word[i]} normal", linestyle="-")
        plt.plot(x, data[:,i*2+1], label=f"{finger_word[i]} tangential", linestyle="--")

    # 添加标题和标签
    plt.title('Left Hand Tactile Feedback')
    plt.xlabel('Time (ms)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.savefig(path)

# 控制过程
def process(frame):
    global ConcateFlag

    if  frame == 10:
        set_ctrl("mot_joint1", 0.05)
        set_ctrl("mot_joint2", -1.5708-0.28)
        set_ctrl("mot_joint5", 0.28)
        set_ctrl("mot_joint6", 1.5708)
        set_ctrl("act_joint1_1", 1.5708)

    elif frame == 1000:
        set_ctrl("mot_joint1", 0.00)
        ConcateFlag = True

    if ConcateFlag:
        all_reached = True
        for i in range(len(current_qpos)):
            target = target_qpos[i]

            # 计算剩余距离
            remaining = target - current_qpos[i]
            if abs(remaining) <= tolerance:
                current_qpos[i] = target
                continue
            else:
                all_reached = False

            # 动态步长
            step = remaining * step_ratio
            current_qpos[i] += step

        Data.ctrl[6:21] = current_qpos
        if all_reached:
            print("所有关节已到达目标位置")
            ConcateFlag = False

    if True:
        user1_id = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_SENSOR, "TS-F-A-1")
        user1_data_id = Model.sensor_adr[user1_id]
        sdata = Data.sensordata[[1,2, 12,13, 23,24, 34,35, 45,46] + user1_data_id]
        Fc_data.append(sdata.copy())

def mujoco_viewer():

    frame = 0
    with viewer.launch_passive(Model, Data) as V:
        while V.is_running():
            step_start = time.time()

            process(frame)
            mujoco.mj_step(Model, Data)     # 更新物理状态和传感器数据

            # 查看器选项的修改示例：每两秒钟切换一次接触点
            with V.lock():
                V.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(Data.time % 2)
            # 获取物理状态的更改，应用扰动，从GUI更新选项
            V.sync()
            # 粗略的计时，相对于挂钟会有漂移
            time_until_next_step = Model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            frame += 1
            if frame == 5000:
                break

if __name__ == "__main__":
    xml_path = "./mujoco_model/DexHand.xml"
    print(f"File exists: {os.path.exists(xml_path)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"File is readable: {os.access(xml_path, os.R_OK)}")

    Model = mujoco.MjModel.from_xml_path(xml_path)
    Data = mujoco.MjData(Model)

    ConcateFlag= False
    Fc_data = []
    mujoco_viewer()
    plt_fc()


