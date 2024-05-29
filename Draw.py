import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics_and_loss(csv_files, metrics_info,subplot_layout, figure_size=(15, 10)):
    plt.figure(figsize=figure_size)

    for i, (metric_name, metric_title) in enumerate(metrics_info):
        plt.subplot(*subplot_layout, i + 1)
        for file_path, name in csv_files:
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == metric_name][0]
            plt.plot(data[column_name], label=name)
        plt.xlabel('Epoch')
        plt.title(metric_title)
        plt.legend()

    plt.tight_layout()
    filename = 'metrics_and_loss_curves.svg'
    plt.savefig(filename,dpi=600)
    plt.show()

    return filename


# Metrics to plot
metrics_info = [
    ('metrics/precision(B)', 'Precision'),
    ('metrics/recall(B)', 'Recall'),
    ('metrics/mAP50(B)', 'mAP at IoU=0.5'),
    ('metrics/mAP50-95(B)', 'mAP for IoU Range 0.5-0.95')
]




# List of CSV files and their corresponding names
csv_files = [
    ('D:/Downloads/YOLOv8/result/result_8_HSFPN/train/exp/results.csv', 'YOLOv8-HSFPN'),
    ('D:/Downloads/YOLOv8/result/result_1_未修改/results.csv', 'YOLOv8'),
    #('D:/Downloads/YOLOv8/result/result_15_slimNeck/train/exp/results.csv', 'YOLOv8-SlimNeck'),
    #('D:/Downloads/YOLOv8/result/result_6_v5/train/exp3/results.csv', 'YOLOv5')
    #('D:/Downloads/YOLOv8/result/result_14_HATHead/train/exp8/results.csv', 'YOLOv5')
]
# Plot the metrics and loss from multiple CSV files
metrics_and_loss_filename = plot_metrics_and_loss(
    csv_files=csv_files,
    metrics_info=metrics_info,
    subplot_layout=(2, len(metrics_info)// 2),
    figure_size=(30, 10)
)
