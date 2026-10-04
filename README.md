# Microduck 截图场景复现

官方模型来源：https://github.com/pollen-robotics/microduck_rl （完整源码保留于 upstream，保留其许可证）。

## 安装与运行

需要 Python 3.11、Git 和可用的 OpenGL 图形环境。克隆本仓库后，在项目目录执行：

```powershell
git submodule update --init --recursive
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

然后打开 http://127.0.0.1:8765 。首次克隆也可以使用 `git clone --recurse-submodules <本仓库地址>`。

官方模型以 Git 子模块固定到 `cb70b792312d559a4da09064d92009079671815f`；其代码、模型和许可证保留在 `upstream`。本地虚拟环境、日志及带机器路径的生成文件不纳入版本控制。

场景：围墙隔板、T 形墙、三面障碍、交错墙。前三个按用户图片中的结构与视觉比例重建，并非原项目源代码或精确尺寸恢复。第三个将起点放在墙外，使进入绿色目标区域前需要主动扫描。

实际使用官方 MJCF/STL 机器人、独立 head_yaw 关节、MuJoCo 渲染与 mj_ray 8×8 墙体射线测距。ToF 最大量程 4 米，未命中为灰色；这是理想化测距，没有硬件噪声。

这是预先展示版本：使用脚本给定底盘轨迹和关节步态，通过 mj_forward 计算姿态，不进行动力学步进，不调用训练权重，也不声称强化学习已经学会避障。左右读数用于可视化比较，路线是预先编排的。

运动表现包含轻微身体侧摆、起伏、步速变化，以及摆头缓入缓出和小幅回调，均为有界的脚本效果。当前仓库提供主动感知行为的场景与可视化预演，不包含研究方法的强化学习训练实现或策略权重。

验证：`.venv\Scripts\python.exe demo.py --verify`。输出在 output 中，包括场景 MJCF、截图及四条路线的完成/几何间隙/扫描记录。
