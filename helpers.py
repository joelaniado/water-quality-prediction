from sklearn.metrics import classification_report
from scikitplot.metrics import plot_roc
from scikitplot.metrics import plot_confusion_matrix
from scikitplot.metrics import plot_precision_recall
from scikitplot.metrics import plot_ks_statistic

import matplotlib.pyplot as plt


class metrics():
    def __init__(self, true, pred, prob=None):
        self.true_label = true
        self.pred_label = pred
        self.prob_label = prob

    def test(self):
        print('test2')

    def report(self):
        print(classification_report(self.true_label, self.pred_label))

    def roc(self):
        if self.prob_label is not None:
            fig, ax = plt.subplots()
            plot_roc(self.true_label, self.prob_label, ax=ax)
            plt.show()

    def confusion(self, norm=False):
        plot_confusion_matrix(self.true_label, self.pred_label, normalize=norm)
        plt.show()

    def PR_curve(self):
        if self.prob_label is not None:
            plot_precision_recall(self.true_label,self.prob_label)
            plt.show()

    def ks(self):
        if self.prob_label is not None:
            plot_ks_statistic(self.true_label,self.prob_label)
            plt.show()



