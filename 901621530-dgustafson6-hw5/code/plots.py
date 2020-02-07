import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	# Citation for loss curves https://github.com/ast0414/CSE6250BDH-LAB-DL/blob/master/3_RNN.ipynb
	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	#plt.show()
	plt.savefig('loss.png')

	plt.figure()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation')
	plt.ylabel('Accuracy')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	#plt.show()
	plt.savefig('acc.png')



def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
    # Citation for confusion matrix code: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    y_true,y_pred = [],[]
    for i in results:
    	y_true.append(i[0])
    	y_pred.append(i[1])
    results = confusion_matrix(y_true,y_pred)
    cmap=plt.cm.Blues
    fig, ax = plt.subplots(2)
    results = np.array(results)
    dat=[results,results.astype('float') / results.sum(axis=1)[:, np.newaxis]]

    for i in [0,1]:
        im = ax[i].imshow(dat[i], interpolation='nearest', cmap=cmap)

        ax[i].figure.colorbar(im, ax=ax[i])
        ax[i].set(xticks=np.arange(dat[i].shape[1]),
               yticks=np.arange(dat[i].shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title='Absolute Confusion' if i==0 else 'Normalized Confusion',
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        fmt = '.2f' if i==1 else 'd'
        thresh = dat[i].max() / 2.
        for k in range(dat[i].shape[0]):
            for j in range(dat[i].shape[1]):
                ax[i].text(j, k, format(dat[i][k, j], fmt),
                        ha="center", va="center",
                        color="white" if results[k, j] > thresh else "black")

    fig.savefig('confusion.png')
    #plt.show()
