from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

def get_q_color(vals):
    colors = ["red", "black", "blue"]
    i = np.argmax(vals)
    return colors[i]


fig = plt.figure(figsize=(12, 9))

ax1 = fig.add_subplot(311)

i = 10000
q_table = np.load(f"qtables/{i}-qtable.npy")


for x, x_vals in enumerate(q_table):
    for y, y_vals in enumerate(x_vals):
        ax1.scatter(x, y, c=get_q_color(y_vals), marker="o")

        ax1.set_ylabel("red left, black none, blue right")


plt.show()

# import cv2
# import os


# def make_video():
#     # windows:
#     fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#     # Linux:
#     # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#     out = cv2.VideoWriter('qlearn.avi', fourcc, 60.0, (1200, 900))

#     for i in range(0, 14000, 10):
#         img_path = f"qtable_charts/{i}.png"
#         frame = cv2.imread(img_path)
#         out.write(frame)

#     out.release()


# make_video()