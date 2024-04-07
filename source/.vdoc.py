# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import torch 
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from torch.nn import Conv2d, MaxPool2d, Parameter
from torch.nn.functional import relu
from sklearn.metrics import confusion_matrix
#
#
#
import warnings
from sklearn.exceptions import ConvergenceWarning
plt.style.use('seaborn-v0_8-whitegrid')
warnings.simplefilter('ignore', ConvergenceWarning)
#
#
#
train_url = "https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/sign-language-mnist/sign_mnist_train.csv"

test_url = "https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/sign-language-mnist/sign_mnist_test.csv"

df_train = pd.read_csv(train_url)
df_test  = pd.read_csv(test_url)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

n, p = df_train.shape[0], df_train.shape[1] - 1
n_test = df_test.shape[0]
#
#
#
#
#

def prep_data(df): 
    n, p = df.shape[0], df.shape[1] - 1
    y = torch.tensor(df["label"].values)
    X = df.drop(["label"], axis = 1)
    X = torch.tensor(X.values)
    X = torch.reshape(X, (n, 1, 28, 28))
    X = X / 255

    return X, y

X_train, y_train = prep_data(df_train)
X_test, y_test = prep_data(df_test)
#
#
#
#
#
X_train[0, 0]
#
#
#
#
#
#---
plt.imshow(X_train[10, 0], cmap = "Greys_r")
#---
plt.gca().set(title = f"{ALPHABET[y_train[10]]}")
no_ax = plt.gca().axis("off")
#
#
#
#
#
def show_images(X, y, rows, cols, channel = 0):

    fig, axarr = plt.subplots(rows, cols, figsize = (2*cols, 2*rows))
    for i, ax in enumerate(axarr.ravel()):
        ax.imshow(X[i, channel].detach(), cmap = "Greys_r")
        ax.set(title = f"{ALPHABET[y[i]]}")
        ax.axis("off")
    plt.tight_layout()

show_images(X_train, y_train, 5, 5)
#
#
#
#
#
#
#
fig, ax = plt.subplots(1, 1, figsize = (6, 2))
letters, counts = torch.unique(y_train, return_counts = True)
proportions = counts / counts.sum()
proportions

ax.scatter(letters, proportions*100, facecolor = "none", edgecolor = "steelblue")
ax.set_xticks(range(26))
ax.set_xticklabels(list(ALPHABET))
ax.set(xlabel = "Letter", ylabel = "Frequency")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 1))
#
#
#
#
#
#
#
#
#
#---
X_train_flat = X_train.reshape(n, p)
print(X_train_flat.size())
#---
#
#
#
#
#
#---
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
f = LR.fit(X_train_flat, y_train)
#---
#
#
#
#
#
#---
LR.score(X_train_flat, y_train)
#---
#
#
#
#
#
#---
X_test_flat = X_test.reshape(n_test, p)
LR.score(X_test_flat, y_test)
#---
#
#
#
#
#
#
#
def vectorization_experiment(pipeline = lambda x: x, return_confusion_matrix = False):
    X_train_transformed = pipeline(X_train)
    X_train_flat = X_train_transformed.flatten(start_dim = 1)
    print(f"Number of features = {X_train_flat.size(1)}")

    LR = LogisticRegression() 
    LR.fit(X_train_flat, y_train)
    print(f"Training accuracy = {LR.score(X_train_flat, y_train):.2f}")

    X_test_transformed = pipeline(X_test) 
    X_test_flat = X_test_transformed.flatten(start_dim = 1)
    print(f"Testing accuracy  = {LR.score(X_test_flat, y_test):.2f}")

    if return_confusion_matrix: 
        y_test_pred = LR.predict(X_test_flat)
        return confusion_matrix(y_test, y_test_pred, normalize = "true")
#
#
#
#
vectorization_experiment() # same experiment as above
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#---
vertical = torch.tensor([0, 0, 5, 0, 0]).repeat(5, 1) - 1.0
vertical
#---
#
#
#
#
#
diag1    = torch.eye(5)*5 - 1
horizontal = torch.transpose(vertical, 1, 0)
diag2    = diag1.flip(1)

fig, ax = plt.subplots(1, 4)
for i, kernel in enumerate([vertical, horizontal, diag1, diag2]):
    ax[i].imshow(kernel, vmin = -1.5, vmax = 2)
    ax[i].axis("off")
    ax[i].set(title = f'{["Vertical", "Horizontal", "Diagonal Down", "Diagonal Up"][i]}')
#
#
#
#
#
def apply_convolutions(X): 

    # this is actually a neural network layer -- we'll learn how to use these
    # in that context soon 
    conv1 = Conv2d(1, 4, 5)

    # need to disable gradients for this layer
    for p in conv1.parameters():
        p.requires_grad = False

    # replace kernels in layer with our custom ones
    conv1.weight[0, 0] = Parameter(vertical)
    conv1.weight[1, 0] = Parameter(horizontal)
    conv1.weight[2, 0] = Parameter(diag1)
    conv1.weight[3, 0] = Parameter(diag2)

    # apply to input data and disable gradients
    return conv1(X).detach()
#
#
#
#
#
#---
X_train_convd = apply_convolutions(X_train)
#---
#
#
#
#
#
#---
X_train_convd.size()
#---
#
#
#
#

def kernel_viz(pipeline):

    fig, ax = plt.subplots(5, 5, figsize = (8, 8))

    X_convd = pipeline(X_train)

    for i in range(5): 
        for j in range(5):
            if i == 0: 
                ax[i,j].imshow(X_train[j, 0])
            
            else: 
                ax[i, j].imshow(X_convd[j,i-1])
            
            ax[i,j].tick_params(
                        axis='both',      
                        which='both',     
                        bottom=False,     
                        left=False,
                        right=False,         
                        labelbottom=False, 
                        labelleft=False)
            ax[i,j].grid(False)
            ax[i, 0].set(ylabel = ["Original", "Vertical", "Horizontal", "Diag Down", "Diag Up"][i])

kernel_viz(apply_convolutions)
#
#
#
#
#
#
#
#---
vectorization_experiment(apply_convolutions)
#---
#
#
#
#
#
#
#
#
#
#
#
z = torch.linspace(-1, 1, 101)
f = relu(x)
plt.plot(x, y, color = "slategrey")
labs = plt.gca().set(xlabel = r"$z$",ylabel = r"$\mathrm{ReLU}(z)$")
#
#
#
#
#
#---
pipeline = lambda x: relu(apply_convolutions(x))
kernel_viz(pipeline)
#---
#
#
#
#
#
#---
vectorization_experiment(pipeline)
#---
#
#
#
#
#
#
#
#
#
#
#
#
#
#---
pipeline = lambda x: MaxPool2d(4,4)(relu(apply_convolutions(x)))
kernel_viz(pipeline)
#---
#
#
#
#
#
#---
vectorization_experiment(pipeline)
#---
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
cm = vectorization_experiment(pipeline, return_confusion_matrix = True)
cm
#
#
#
quote = "Darkness cannot drive out darkness: only light can do that. Hate cannot drive out hate: only love can do that."

alphabet_map = {a : i for i, a in enumerate(alphabet)}

for i, letter in enumerate(quote.upper()): 
    if letter.upper() not in alphabet:
        continue
    j = alphabet_map[letter]
    row = cm[j,:]
    print(torch.multinomial(torch.Tensor(row), 1))


#
#
#
#
#
