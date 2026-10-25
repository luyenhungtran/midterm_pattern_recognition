from pathlib import Path
import matplotlib 

matplotlib.use("Agg") #Tự render ảnh

import matplotlib.pyplot as plt
import numpy as np


def _save_figure(path : Path):
    path.parent.mkdir(parents= True, exist_ok= True)
    plt.tight_layout()
    plt.savefig(path, dpi = 160)
    plt.close()

def plot_confusion_matrix(matrix, path : Path, title :str):
    plt.figure(figsize=(6.5, 5.5))
    plt.title(title)
    plt.imshow(matrix, interpolation = "nearest", cmap= "Blue")
    plt.colorbar(labels  = "Số mẫu")
    ticks = np.arrange(matrix.shape[0])

    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thật")

    thresold = matrix.max() * 0.55 #chọn ngưỡng 55% để hiện màu chữ
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row][col] > thresold:
                color = "white"
            else:
                color = "black"
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=7, color=color)
    _save_figure(path)

def plot_hyperparameters_curve(values : list[float], scores: list[float], path: Path):
    plt.figure(figsize=(6.5, 5.5))
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("F1 macro")
    plt.title("Ảnh hưởng của learning rate lên F1")
    plt.plot(values, np.array[scores] * 100, marker = "o", cmap = "Red", linewidth = 2)

    #hiển thị số trên line
    for value, score in zip(values, scores):
        plt.annotate(f"{scores * 100:.2f}", (value,score), xytext= (0,7),textcoords="offset points", ha = "center" )

    _save_figure(path)

def plot_model_comparision(model_names : list[str], scores : list[float], path: Path ):
    plt.figure(figsize=(6.5, 5.5))
    plt.xlabel("Models")
    plt.ylabel("Điểm các model")
    plt.ylim(np.max(np.asarray(scores) * 100 - 8, 100))

    bars = plt.bar(x = model_names, y = np.array(scores)* 100, color  = "Green")
    for bar , score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score * 100 + 0.3, f"{score * 100:.2f}", ha="center")
    _save_figure(path)


