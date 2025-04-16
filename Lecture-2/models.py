# This code is adapted from https://github.com/rasbt/machine-learning-book/ by Sebastian Raschka
# Licensed under the MIT License

# Generic imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ------------------------------------------------------------------------------------------------
# ****************************************** The Perceptron **************************************
# ------------------------------------------------------------------------------------------------
class Perceptron:
    """
    Perceptron classifier for binary classification tasks.

    This model learns a linear decision boundary to separate two classes
    using the Perceptron learning algorithm. It updates weights incrementally
    whenever a misclassification occurs.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0). Controls the size of the weight update steps.
    n_epochs : int
        Number of passes (epochs) over the entire training dataset.
    random_state : int
        Seed for the random number generator used to initialize weights.

    Attributes
    ----------
    w_ : 1d-array
        Weights associated with the input features, learned during training.
    b_ : float
        Bias term learned during training.
    errors_ : list of int
        Tracks the number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta=0.01, n_epochs=50, random_state=1):
        self.eta = eta
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the Perceptron model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix, where each row is a training sample and each column is a feature.
        y : array-like of shape (n_samples,)
            Target labels, expected to be binary (0 or 1).

        Returns
        -------
        self : object
            Fitted model instance.
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.errors_ = []

        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate the net input (weighted sum plus bias).

        Parameters
        ----------
        X : array-like of shape (n_features,) or (n_samples, n_features)
            Input vector(s) for which to calculate the net input.

        Returns
        -------
        float or array-like
            Linear combination of inputs and weights plus bias.
        """
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """
        Predict the class label based on the net input.

        Applies the unit step function: returns 1 if the net input is >= 0, else returns 0.

        Parameters
        ----------
        X : array-like of shape (n_features,) or (n_samples, n_features)
            Input vector(s) to classify.

        Returns
        -------
        int or array-like
            Predicted class label(s).
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# ------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
# ******************************** Decision Boundary Plotting Tool *******************************
# ------------------------------------------------------------------------------------------------
def plot_decision_regions(X, y, classifier, resolution=0.02, class_names=None):
    """
    Plot decision boundaries for a classifier in a 2D feature space.

    This function visualizes the decision regions of a classifier, allowing the user
    to inspect how the model separates different classes in the input space. It supports
    binary and multiclass classification problems, and allows for custom class names.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Feature matrix with exactly two features (columns) for visualization.
    y : array-like of shape (n_samples,)
        Target class labels.
    classifier : object
        Trained classifier with a .predict() method.
    resolution : float, optional (default=0.02)
        Grid resolution for plotting the decision surface.
    class_names : list of str, optional
        Custom names for classes to display in the legend. If None, uses class integers.

    Returns
    -------
    None
    """

    # Setup marker generator and color map
    markers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )

    grid_points = np.array([xx1.ravel(), xx2.ravel()]).T
    lab = classifier.predict(grid_points)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        label = class_names[idx] if class_names is not None else f"Class {cl}"
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=label,
            edgecolor="black",
        )


# ------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
# ************************************** ADAptive LInear NEuron **********************************
# ------------------------------------------------------------------------------------------------
class AdalineGD:
    """
    ADAptive LInear NEuron classifier using batch gradient descent.

    This implementation of Adaline minimizes the mean squared error (MSE)
    between predicted and actual class labels using linear activation by default.
    Users may override the activation function for more advanced use cases.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Number of passes (epochs) over the training dataset.
    random_state : int
        Random number generator seed for reproducible weight initialization.
    activation : callable, optional (default: identity function)
        Custom activation function. Must accept a NumPy array and return a NumPy array.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : float
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss value for each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1, activation=None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.activation_fn = activation if activation is not None else self._identity

    def fit(self, X, y):
        """
        Fit training data using batch gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Trained model instance.
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation_fn(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        """
        Calculate the net input as the linear combination of weights and inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        net_input : array-like of shape (n_samples,)
            The result of the dot product plus bias.
        """
        return np.dot(X, self.w_) + self.b_

    def _identity(self, X):
        """
        Default identity activation function.

        Parameters
        ----------
        X : array-like
            Input array.

        Returns
        -------
        X : array-like
            Same as input.
        """
        return X

    def predict(self, X):
        """
        Predict binary class labels based on activation threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        labels : array-like of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        return np.where(self.activation_fn(self.net_input(X)) >= 0.5, 1, 0)


# ------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
# ******************************* ADAptive LInear NEuron with SGD ********************************
# ------------------------------------------------------------------------------------------------
class AdalineSGD:
    """
    ADAptive LInear NEuron classifier using stochastic gradient descent.

    This implementation updates weights incrementally after each training sample,
    enabling online learning and faster updates. By default, it uses a linear (identity)
    activation function but allows overriding with a custom function.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Number of passes (epochs) over the training dataset.
    shuffle : bool, default=True
        If True, shuffles the training data every epoch to prevent cycles.
    random_state : int, optional
        Seed for random weight initialization and shuffling.
    activation : callable, optional (default: identity function)
        Custom activation function. Must accept a NumPy array and return a NumPy array.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : float
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss value averaged over all training examples in each epoch.
    """

    def __init__(
        self, eta=0.01, n_iter=10, shuffle=True, random_state=None, activation=None
    ):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.activation_fn = activation if activation is not None else self._identity
        self.w_initialized = False

    def fit(self, X, y):
        """
        Fit training data using stochastic gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Trained model instance.
        """
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        """
        Perform one iteration of training on a batch without reinitializing weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Partially trained model.
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """
        Shuffle training data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.

        Returns
        -------
        X, y : tuple of array-like
            Shuffled feature matrix and target vector.
        """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """
        Initialize weights to small random values.

        Parameters
        ----------
        m : int
            Number of features.
        """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.0)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """
        Apply Adaline learning rule to update weights and bias.

        Parameters
        ----------
        xi : array-like of shape (n_features,)
            Feature vector.
        target : int or float
            True class label.

        Returns
        -------
        loss : float
            Squared error loss for the given example.
        """
        output = self.activation_fn(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi * error
        self.b_ += self.eta * 2.0 * error
        return error**2

    def net_input(self, X):
        """
        Calculate the net input as the linear combination of weights and inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        net_input : array-like or float
            Linear combination plus bias.
        """
        return np.dot(X, self.w_) + self.b_

    def _identity(self, X):
        """
        Default identity activation function.

        Parameters
        ----------
        X : array-like
            Input array.

        Returns
        -------
        X : array-like
            Same as input.
        """
        return X

    def predict(self, X):
        """
        Predict binary class labels based on activation threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        labels : array-like of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        return np.where(self.activation_fn(self.net_input(X)) >= 0.5, 1, 0)


# ------------------------------------------------------------------------------------------------
