#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# In[2]:


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


# In[3]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version = 1, cache = True)
mnist.target = mnist.target.astype(np.int8)
sort_by_target(mnist)


# In[4]:


print(mnist.data.shape)


# In[5]:


X, y = mnist["data"], mnist["target"]
print(X.shape, y.shape)


# In[6]:


some_digit = X[36000]
print(some_digit)


# In[7]:


some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis('off')
#save_fig("some_digit_plot")
plt.show()


# In[8]:


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary)
    plt.axis('off')
    plt.show()


# In[9]:


print(y[36000])


# In[10]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[11]:


shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[12]:


y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
print(y_train_5)


# In[13]:


from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
clf.fit(X_train, y_train_5)


# In[14]:


clf.predict([some_digit])


# In[15]:


from sklearn.model_selection import cross_val_score
cross_val_score(clf, X_train, y_train_5, cv=3, scoring="accuracy")


# In[16]:


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf, X_train, y_train_5, cv=3)


# In[17]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)


# In[18]:


y_train_perfect_predictions = y_train_5


# In[19]:


confusion_matrix(y_train_5, y_train_perfect_predictions)


# In[20]:


from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_5, y_train_pred))
print(3768/(3768+746))


# In[21]:


print(recall_score(y_train_5, y_train_pred))
print(3768/(3768+1653))


# In[22]:


from sklearn.metrics import f1_score
print(f1_score(y_train_5, y_train_pred))
print(3768/(3768+(746+1653)/2))


# In[32]:


y_scores = clf.decision_function([some_digit])
print(y_scores)


# In[34]:


y_scores = cross_val_predict(clf, X_train, y_train_5, cv=3,
                             method="decision_function")


# In[35]:


from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[37]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision', linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g--', label='recall', linewidth=2)
    plt.xlabel('threshold', fontsize=16)
    plt.legend(loc='upper left', fontsize=16)
    plt.ylim([0,1])
    
plt.figure(figsize=(8,4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000,700000])
plt.show()


# In[39]:


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.show()


# In[40]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# In[45]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False positive rate', fontsize=16)
    plt.ylabel('True positive rate', fontsize=16)
    
plt.figure(figsize=(8,6))
plot_roc_curve(fpr, tpr)
plt.show()


# In[46]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


# In[47]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')


# In[52]:


y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# In[54]:


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right", fontsize=16)
plt.show()


# In[55]:


roc_auc_score(y_train_5, y_scores_forest)


# In[66]:


y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
precision_score(y_train_5, y_train_pred_forest)


# In[67]:


recall_score(y_train_5, y_train_pred_forest, average='weighted')


# In[ ]:





# In[ ]:




