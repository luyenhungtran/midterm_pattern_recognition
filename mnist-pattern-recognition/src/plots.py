from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(matrix, title: str): 
    plt.figure(figsize=(6.5, 5.5))
    plt.title(title)
 
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(label="Số mẫu")
    
    ticks = np.arange(matrix.shape[0])

    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thật")

    threshold = matrix.max() * 0.55  # Chọn ngưỡng 55% để hiện màu chữ
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row][col] > threshold:
                color = "white"
            else:
                color = "black"
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=7, color=color)
    
    plt.show()


def plot_hyperparameters_curve(values: list[float], scores: list[float]):
    plt.figure(figsize=(6.5, 5.5))
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("F1 macro")
    plt.title("Ảnh hưởng của learning rate lên F1")
    
    plt.plot(values, np.array(scores) * 100, marker="o", color="red", linewidth=2)

    # Hiển thị số trên đường biểu diễn 
    for value, score in zip(values, scores):
        plt.annotate(f"{score * 100:.2f}", (value, score * 100), xytext=(0, 7), textcoords="offset points", ha="center")

    plt.show()


def plot_model_comparision(model_names: list[str], scores: list[float]):
    plt.figure(figsize=(6.5, 5.5))
    plt.xlabel("Models")
    plt.ylabel("Điểm các model")

    # Sửa plt.ylim: giới hạn dưới và trên truyền đúng cú pháp tuple/list
    scores_array = np.asarray(scores) * 100
    plt.ylim(np.max(scores_array) - 8, 100)

    bars = plt.bar(x=model_names, height=scores_array, color="green")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score * 100 + 0.3, f"{score * 100:.2f}", ha="center")
    
    plt.show()


def plot_model_time(times: list[float], model_names: list[str]):
    plt.figure(figsize=(6.5, 5.5))
    plt.xlabel("Model")
    plt.ylabel("Thời gian chạy")
    
    bars = plt.bar(x=model_names, height=times, color="blue")
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, time, f"{time}", ha="center")
    
    plt.show()