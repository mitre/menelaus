import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import svm

# read in Circle dataset
# assumes the script is being run from the root directory.
# TODO: change this path back after done testing
# df = pd.read_csv(
#     os.path.join(
#         "..", "..", "src", "menelaus", "tools", "artifacts", "dataCircleGSev3Sp3Train.csv"
#     ),
#     usecols=[0, 1, 2],
#     names=["var1", "var2", "y"],
# )
# drift_start, drift_end = 1000, 1250
# training_size = 500

# # Set up classifier: train on first training_size rows
# X = df.loc[0:training_size, ["var1", "var2"]]
# Y = df.loc[0:training_size, "y"]

# we create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# figure number
fignum = 1

# fit the model
for name, penalty in (("unreg", 1), ("reg", 0.05)):

    clf = svm.SVC(kernel="linear", C=penalty)
    clf.fit(X, Y)

    # get the separating hyperplane
    w = clf.coef_[0]
    print("w", w)
    a = -w[0] / w[1]
    print("a", a)
    xx = np.linspace(-5, 5)
    print("xx", xx)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    print("yy", yy)

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_**2))
    print("margin", margin)
    yy_down = yy - np.sqrt(1 + a**2) * margin
    print("yy_down", yy_down)
    yy_up = yy + np.sqrt(1 + a**2) * margin
    print("yy_up", yy_up)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, "k-")
    plt.plot(xx, yy_down, "k--")
    plt.plot(xx, yy_up, "k--")

    plt.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=80,
        facecolors="none",
        zorder=10,
        edgecolors="k",
        cmap=cm.get_cmap("RdBu"),
    )
    plt.scatter(
        X[:, 0], X[:, 1], c=Y, zorder=10, cmap=cm.get_cmap("RdBu"), edgecolors="k"
    )

    plt.axis("tight")
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Put the result into a contour plot
    plt.contourf(XX, YY, Z, cmap=cm.get_cmap("RdBu"), alpha=0.5, linestyles=["-"])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1

plt.show()