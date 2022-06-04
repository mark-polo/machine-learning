import matplotlib.pyplot as plt


class matplot_visualisation:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def visualisation(self):
        fig, ax = plt.subplots(1,1)
        ax.scatter(self.X, self.Y)
        plt.show()