import numpy as np
import matplotlib.pyplot as plt

def plot_frontiers(X,y,model,feature_names,target_names,nclasses=3,bcolors="RdBu",pcolors='ryb'):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                      np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=bcolors)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])

    # Plot the training points
    for i, color in zip(range(nclasses), pcolors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=target_names[i],
                    edgecolor='black', s=20)
    plt.legend()
    plt.show()
  
 