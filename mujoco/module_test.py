import time
import mujoco
from mujoco import viewer
import numpy as np
import os
import matplotlib.pyplot as plt
from mjcb_sensor.linux import TSensor

TSensor.register_sensor_callback()      # 注册回调

def set_ctrl(name, value):
    id = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    Data.ctrl[id] = value

# 绘制折线
def plt_fc():
    current_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(current_path, "./sensor_data/module.png")

    plt.figure(figsize=(12,8))
    data = np.array(Fc_data)
    x = [i for i in range(len(Fc_data))]

    plt.plot(x, data[:,0], label="th normal", linestyle="-")
    # plt.plot(x, data[:,1], label="th tangential", linestyle="--")

    # 添加标题和标签
    plt.title('Module Tactile Feedback')
    plt.xlabel('Time (ms)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.savefig(path)

def write_csv(data):

    data = np.array(data)
    # 定义列名（逗号分隔）
    header = "th_n, th_t, th_d"

    # 保存到CSV文件
    with open("./sensor_data/data.csv", mode="w", encoding="utf-8") as f:
        f.write(header + "\n")
        np.savetxt(f, data, delimiter=",", fmt="%f")

    print("保存成功！")


def mujoco_viewer():

    frame = 0
    with viewer.launch_passive(Model, Data) as V:
        while V.is_running():
            step_start = time.time()

            mujoco.mj_step(Model, Data)     # 更新物理状态和传感器数据

            user1_id = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_SENSOR, "TS-F-A-1")
            user1_data_id = Model.sensor_adr[user1_id]
            sdata = Data.sensordata[[1,2] + user1_data_id]
            Fc_data.append(sdata.copy())

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
            if frame == 1000:
                break

if __name__ == "__main__":
    xml_path = "./mujoco_model/TS-F-A.xml"
    print(f"File exists: {os.path.exists(xml_path)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"File is readable: {os.access(xml_path, os.R_OK)}")

    Model = mujoco.MjModel.from_xml_path(xml_path)
    Data = mujoco.MjData(Model)

    Fc_data = []
    mujoco_viewer()
    plt_fc()


