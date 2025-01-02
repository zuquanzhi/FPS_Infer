import matplotlib.pyplot as plt

# 读取 FPS 数据文件
file_path = "fps_output.txt"  # 替换为您的文件路径
infer_times = []
fps_values = []

# 读取数据
with open(file_path, 'r') as file:
    for line in file:
        if "Infer time(ms):" in line and "FPS:" in line:
            # 提取 Infer time 和 FPS 值
            parts = line.split(", ")
            infer_time = float(parts[0].split(": ")[1])
            fps = float(parts[1].split(": ")[1])
            
            infer_times.append(infer_time)
            fps_values.append(fps)

# 绘制图表
plt.figure(figsize=(10, 5))

# 绘制 FPS 曲线
plt.subplot(1, 2, 1)
plt.plot(fps_values, color='blue', label='FPS')
plt.title('FPS over Time')
plt.xlabel('Iteration')
plt.ylabel('FPS')
plt.grid(True)

# 绘制推理时间（Infer Time）曲线
plt.subplot(1, 2, 2)
plt.plot(infer_times, color='red', label='Infer Time (ms)')
plt.title('Infer Time over Time')
plt.xlabel('Iteration')
plt.ylabel('Infer Time (ms)')
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.show()
