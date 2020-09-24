#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

import time
import random
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.special import logsumexp
from pylogit import create_choice_model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, regularizers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import convert_to_tensor, GradientTape

from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin


class RACBoost(BaseEstimator, ClassifierMixin):
    """
    Regularised Augmented Conditional Boosting Model

    Parameters
    ----------
    alt_id_col :str.
        Denote the column in data X which contains the alternative
        identifiers for each row.

    obs_id_col : str.
        Denote the column in data X which contains the observation
        identifiers for each row.

    choice_col : str.
        Denote the column in data y which contains the ones and zeros that
        denote whether or not the given row corresponds to the chosen
        alternative for the given individual.

    loss : object, default : Regularised Conditional Logit Loss Function
        Loss Function to be optimized.

    reg_lambda_l1 : float, default=0.0
        L1 Regularisatio nterm of weight of neural net.

    reg_lambda_l2 : float, default=0.0
        L2 Regularisatio nterm of weight of neural net.

    nn_complexity : float, default=0.0
        Regularisation term on the complexity of the nerual net.

    reg_leaf : int, default=0
        Regularisation term on the terminal nodes in regression tree.

    ignore_features : list / str, default='dominant'
        Features to exclude from the training process

    nn_count : int, default='n_features'
        Number of neurons in the first hidden layer of the neural net in the
        feature augmentation step

    nn_shrink : float, default=0.3
        Scaler for number of neurons in subsequent hidden layers of the neural
        net in the feature augmentation step

    nn_size : float, default=1
        Scaler for nn_count between iterations

    max_depth : int, default=3
        Maximum depth of the individual regression tree base learners.
        The maximum depth limits the number of nodes in the tree

    learning_rate : float, default=0.1
        Boosting learning rate that shrinks the contribution of
        each regression tree base learners.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int, default=500
        The number of boosting stages to perform.

    post_lambda : float, default=0.0
        L1 regularisation term on the base learners in the post-processing step

    row_subsample : float, default=1.0
        The fraction of training instances used in feature augmentation step
        If smaller than 1.0 this results in Stochastic Gradient Boosting.

    col_subsample : float, default=1.0
        The fraction of columns used in the feature augmentation step

    random_state : int, default = None
        Controls the random seed given to each regression tree base learners.
        Controls the random seed given to each neural net feature augmentor.

    verbose : Bool, defualt=True
        Enable verbose output.
        If True then prints progress and performance

    Attributes
    ----------
    features_list_ : list
        List of features from column names in dataset.

    n_features_ : int
        Nuumber of features in the dataset

    dominant_feature_ : str
        Feature Name of the dominant feature identified

    loss_ : LossFunction
        The concrete ititialised LossFunction object.

    train_score_ : ndarray of training loss function score

    feature_aug_ : ndarray of Keras Neural Nets, shape(n_estimators,1)
        The collection of fitted feature augmentation Keras Neural Nets

    estimators_ : ndarray of DecisionTreeRegressor, shape (n_estimators,1)
        The collection of fitted base learners.

    decision_function :

    """

    def __init__(
            self,
            alt_id_col,
            obs_id_col,
            choice_col,
            loss=None,
            reg_lambda_l1=0.0,
            reg_lambda_l2=0.0,
            nn_complexity=0.0,
            reg_leaf=0.0,
            ignore_features='dominant',
            nn_count='n_features',
            nn_shrink=0.3,
            nn_size=1.0,
            max_depth=3,
            learning_rate=0.1,
            n_estimators=500,
            post_lambda=0.0,
            row_subsample=1.0,
            col_subsample=1.0,
            random_state=None,
            verbose=1):

        self.alt_id_col = alt_id_col
        self.obs_id_col = obs_id_col
        self.choice_col = choice_col
        self.loss = loss

        self.reg_lambda_l1 = reg_lambda_l1
        self.reg_lambda_l2 = reg_lambda_l2
        self.nn_complexity = nn_complexity
        self.reg_leaf = reg_leaf
        self.ignore_features = ignore_features
        self.nn_count = nn_count
        self.nn_shrink = nn_shrink
        self.nn_size = nn_size
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.post_lambda = post_lambda
        self.row_subsample = row_subsample
        self.col_subsample = col_subsample

        self.random_state = random_state
        self.verbose = verbose

        return None

    def fit(self, X, y):
        """
        Iteratively fit the stages of the Gradient Boosting Model
        For each iteration, it prints the progress score if verbose = 1

        Parameters
        ----------
        X : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, n_features + 2)
            the dataframe should be in long format for the discrete choice model
            The input samples.
        y : Pandas DataFrame with alt_id_col, obs_id_col and choice_col
            of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
            The targets.
        Returns
        -------
        self : object

        """
        # Initiate Verbose Reporter
        if self.verbose:
           self.verbose_reporter = VerboseReporter()

        # Base Attributes
        self.X_base_cols_ = [self.alt_id_col, self.obs_id_col]
        self.y_base_cols_ = [self.alt_id_col, self.obs_id_col, self.choice_col]
        self.individuals_ = list(X.loc[:, self.obs_id_col].unique())
        self.n_individuals_ = len(self.individuals_)
        self.features_list_ = list(X.columns.drop(self.X_base_cols_))
        self.n_features_ = len(self.features_list_)

        # Check input
        self._validate_data(X, y)

        # Check Parameters
        self._check_params()

        # Step 1 : Initialize model
        raw_predictions = self._initialisation(X, y)

        # Iteratively fits the boosting stages
        for i in range(self.n_estimators):

            # A copy is passed to negative_gradient() because raw_predictions
            # is updated at the end of the loop in Step 5
            raw_predictions_copy = raw_predictions.copy()

            """
            Step 2 : Computeing Negative Residual
            """
            residuals = self.loss_._negative_gradient(y, raw_predictions_copy)

            """
            Step 3 : Stage-wise Neural Net Feature Augmentation
            """
            features, nn, nn_complexity_measure, col_idx = \
                self._feature_augmentation(X, residuals)

            """
            Step 4 : Fitting Base Learner to Negative Gradient
            """
            tree = self._fit_base_learner(features, residuals)

            """
            Step 5 : Line Search Updating terminal regions
            """
            terminal_values = self._line_search(tree.tree_, features, \
                residuals, raw_predictions, nn_complexity_measure)

            """
            Step 6 : Update via Shrinkage
            """
            raw_predictions = self._update(i, terminal_values, raw_predictions)

            """
            Step 7 : Save Iteration
            """
            self._save_iteration(i, nn, tree, col_idx, y, raw_predictions)

        """
        Step 8 : LASSO Post-Processing
        """
        self._post_processing(y)

        return self

    def _validate_data(self, X, y):
        """
        Validate Input data : Datatype and existance of alt_id_col, obs_id_col
        Parameters
        ----------
        X : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, n_features + 2)
            the dataframe should be in long format for the discrete choice model
            The input samples.

        y : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
            The targets.
        """
        # Checks whether X amd y are Pandas DataFrames
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame but "
                            f"was {type(X)}")

        if not isinstance(y, pd.DataFrame):
            raise TypeError("y must be a pd.DataFrame but "
                            f"was {type(y)}")

        # Checks whether alt_id_col and obs_id_col is in Dataframes X and y
        problem_cols_X = [
            col for col in self.X_base_cols_ if col not in X.columns]
        problem_cols_y = [
            col for col in self.y_base_cols_ if col not in y.columns]

        if problem_cols_X != []:
            raise ValueError("The following columns in are not in "
                             f"X.columns : {problem_cols_X}")

        if problem_cols_y != []:
            raise ValueError("The following columns in are not in "
                             f"X.columns : {problem_cols_y}")

        return None

    def _check_params(self):
        """
        Check validity of parameters and raise ValueError if not valid.
        """

        if self.reg_lambda_l1 < 0.0:
            raise ValueError("reg_lambda_l1 must be non-negative but "
                             f"was {self.reg_lambda_l1}")

        if self.reg_lambda_l2 < 0.0:
            raise ValueError("reg_lambda_l2 must be non-negative but "
                             f"was {self.reg_lambda_l2}")

        if type(self.nn_complexity) != np.float64 and type(self.nn_complexity) != float:
            raise ValueError("nn_complexity must be a floating point nuumber but "
                             f"was {self.nn_complexity}")

        if self.reg_leaf < 0.0:
            raise ValueError("reg_leaf must be non-negative but "
                             f"was {self.reg_leaf}")

        if self.ignore_features != 'dominant':
            if not isinstance(self.ignore_features, list):
                raise TypeError("ignore_feature must be a list but "
                                f"was {type(self.ignore_features)}")

            # Check whether all ignored features are in the feature list
            problem_cols = [
                col for col in self.ignore_features if col not in self.features_list_]
            if problem_cols != []:
                raise ValueError("The following columns in are not in "
                                 f"feature list : {problem_cols}")

        if self.nn_count != 'n_features':
            if self.nn_count < 1:
                raise ValueError(
                    "nn_count must be greater than 0 or 'n_features' but "
                    f"was {self.nn_count}")

        if not (0.0 < self.nn_shrink < 1.0):
            raise ValueError("nn_shrink must be in (0,1) but "
                             f"was {self.nn_shrink}")

        if self.nn_size <= 0.0:
            raise ValueError("nn_size must be greater than 0 but "
                             f"was {self.nn_size}")

        if self.max_depth < 1:
            raise ValueError("max_depth must be greater than 0 but "
                             f"was {self.max_depth}")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             f"was {self.learning_rate}")

        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1 but "
                             f"was {self.n_estimators}")

        if self.post_lambda < 0.0:
            raise ValueError("post_lambda must be non-negative but "
                             f"was {self.post_lambda}")

        if not (0.0 < self.row_subsample <= 1.0):
            raise ValueError("row_subsample must be in (0,1] but "
                             f"was {self.row_subsample}")

        if not (0.0 < self.col_subsample <= 1.0):
            raise ValueError("col_subsample must be in (0,1] but "
                             f"was {self.col_subsample}")

        return None

    def _initialisation(self, X, y):
        """
        Step 1 : Initialize model state by identifying the dominant feature
        Parameters
        ----------
        X : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, n_features + 2)
            the dataframe should be in long format for the discrete choice model
            The input samples.
        y : Pandas DataFrame with alt_id_col, obs_id_col and choice_col
            of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
            The targets.
        Returns
        -------
        raw_predictions : Pandas DataFrame with alt_id_col, obs_id_col,
            utility (estimated utility function) and utility_obs
            (sum of all utilities in obs) of the shape (n_samples, 4)
            the dataframe should be in long format for the discrete choice model
        """
        # Initial Loss function and Predictior
        self.loss_ = self.loss(
            alt_id_col=self.alt_id_col,
            obs_id_col=self.obs_id_col,
            choice_col=self.choice_col)
        self.init_ = self.loss_.init_estimator()

        self.loss_.init_estimator_warnings('ignore')

        init_set = y.merge(X)

        max_likelihood = -np.inf
        for col in self.features_list_:
            spec = OrderedDict()
            spec[col] = 'all_same'
            zeros = np.zeros(len(spec))
            model = self.init_(data=init_set.loc[:, self.y_base_cols_ + [col]],
                               alt_id_col=self.alt_id_col,
                               obs_id_col=self.obs_id_col,
                               choice_col=self.choice_col,
                               specification=spec,
                               model_type="MNL")
            model.fit_mle(zeros, print_res=False)
            if model.log_likelihood > max_likelihood:
                max_likelihood = model.log_likelihood
                self.dominant_model = model
                self.dominant_feature_ = col

        # Initialise raw_predictions
        raw_predictions = X.loc[:, self.X_base_cols_]
        raw_predictions.loc[:,'utility'] = self.dominant_model.params.values \
            * X.loc[:, self.dominant_feature_]
        raw_predictions.loc[:, 'utility_obs'] = self.loss_._utility_obs(
            raw_predictions)

        self.loss_.init_estimator_warnings('default')

        # Initialise decision functino to store terminal regions
        self.decision_function = raw_predictions.loc[:,
                                            self.X_base_cols_ + ['utility']]
        self.decision_function.columns = self.X_base_cols_ + ['init']

        # Initialise length of train_score_ vector
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)

        # Initialise length of feature_aug_ vector
        self.feature_aug_ = np.empty((self.n_estimators,), dtype=np.object)

        # Initialise length of col_idx_ vector
        self.col_idx_ = np.empty((self.n_estimators,), dtype=np.object)

        # Initialise length of estimators_ vector
        self.estimators_ = np.empty((self.n_estimators,), dtype=np.object)

        # Verbose
        if self.verbose:
            self.verbose_reporter.start()

        return raw_predictions

    def _feature_augmentation(self, X, residuals):
        """
        Step 3 : Stage-wise Neural Net Feature Augmentation
        Parameters
        ----------
        X : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, n_features + 2)
            the dataframe should be in long format for the discrete choice model
            The input samples.
        residuals : Pandas DataFrame with alt_id_col, obs_id_col and residual
            of the shape (n_samples, 3)
            he dataframe should be in long format for the discrete choice model
        Returns
        -------
        features : Pandas DataFrame without any id of the shape (n_samples, #nn_features)
            the dataframe should be in long format for the discrete choice model
        nn : Keras Neural Net model
            A fitted Keras Neural Net Model
        """

        """
        Subsampling
        """
        # Random state is not set in sampling to allow for differernt samples / features accross different iterations
        # Subsampling rows
        if self.row_subsample < 1:
            # Number of individuals / samples
            num_samples = int(self.n_individuals_ * self.row_subsample)

            # Sampling obs_id_col
            row_idx = random.sample(self.individuals_, num_samples)
        else:
            row_idx = self.individuals_

        # Subsampling columns
        if self.col_subsample < 1:

            # Number of columns
            num_cols = int(self.n_features_ * self.col_subsample)

            if self.ignore_features != []:
                if self.ignore_features == 'dominant':
                    # Sampling features without the dominant term
                    col_idx = random.sample(
                        [col for col in self.features_list_ if col != self.dominant_feature_], num_cols)
                else:
                    # Sampling features ignoring a list of features
                    col_idx = random.sample(
                        [col for col in self.features_list_ if col not in self.ignore_features], num_cols)
            else:
                # Sampling features
                col_idx = random.sample(self.features_list_, num_cols)
        else:
            col_idx = self.features_list_

        subsampled_X = X.loc[X[self.obs_id_col].isin(row_idx), col_idx + [self.obs_id_col]]
        subsampled_y = residuals.loc[residuals[self.obs_id_col].isin(row_idx), :]

        """
        Neural Net Model by a Custom Training Loop
        The custom loop is used to incoperate the obs_ids in the loss function
        Also control individuals are grouped together in a batch
        """
        # Formatting dataset
        # Split obs_ids in batches according to obs
        X_columns = [
            col for col in subsampled_X.columns if col not in self.X_base_cols_]
        train_dataset = [(subsampled_X.loc[subsampled_X[self.obs_id_col] == obs,
                                           X_columns].values,
                          subsampled_y.loc[subsampled_y[self.obs_id_col] == obs,
                                           'residual']) for obs in self.individuals_]
        input_dim = len(X_columns)

        # Create model instance
        model, nn_complexity_measure = self._generate_nn_model(input_dim)

        # Initialise an optimizer
        optimizer = Adam()

        # Initialise an Loss Function
        loss_fn = MeanSquaredError(reduction="auto", name="mean_squared_error")

        epochs = 10
        # Fitting the model
        for epoch in range(epochs):

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                # train_step(x_batch_train, y_batch_train)

                # A GradientTape is used to record the operations during each
                # forward pass
                with GradientTape() as tape:

                    # Run the forward pass
                    y_pred = model(x_batch_train, training=True)

                    # Compute the loss function for this batch.
                    # Future Development : This allows for adding regularisation
                    # parameter to change the loss function between epoches
                    y_true = convert_to_tensor(y_batch_train.values.reshape(
                        y_batch_train.size, 1), dtype='float32')
                    loss_value = loss_fn(y_true, y_pred)

                # Use the gradient tape to automatically retrieve the gradients of the
                # trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Getting feature set
        extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        features = extractor.predict(X.loc[:, col_idx].values)

        # Updating neuron counts for next iteration
        self.nn_count *= self.nn_size

        return features, extractor, nn_complexity_measure, col_idx

    def _generate_nn_model(self, input_dim):
        """
        Generate model based on specifications
        Parameters
        ----------
        input_dim : int
        Dimension of input data

        Returns
        -------
        model : keras model object

        nn_complexity_measure : float
            Measure of complexity
        """
        # Define size of Input
        inputs = Input(shape=(input_dim,))

        # Layer 1 Neuron Count
        if self.nn_count == 'n_features':
            neuron_count = self.n_features_
            self.nn_count = neuron_count
        else:
            neuron_count = self.nn_count


        nn_complexity_measure = 0
        # Add Layers
        layer = inputs
        while neuron_count > 30:

            nn_complexity_measure += input_dim * np.log(int(neuron_count))
            layer = Dense(int(neuron_count),activation='relu',
                          kernel_regularizer=regularizers.l1_l2(
                              l1=self.reg_lambda_l1,l2=self.reg_lambda_l2))(layer)

            #Update Input dimension of next layer
            input_dim = int(neuron_count)

            # Shrink neuron count for next layer
            neuron_count *= self.nn_shrink
        # Add final Layer
        output = Dense(1, activation='sigmoid')(layer)
        model = Model(inputs=inputs, outputs=output)

        return model, nn_complexity_measure

    def _fit_base_learner(self, features, residuals):
        """
        Step 4 : Fitting Base Learner to Negative Gradient
        Parameters
        -----------
        X : Pandas DataFrame with features
            of the shape (n_samples, n_features)
            the dataframe should be in long format for the discrete choice model
            The input samples.
        residuals : Pandas DataFrame with alt_id_col, obs_id_col and residual
            of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
        Returns
        -------
        tree : trained sklearn tree object
        """
        # Init Regression Tree
        tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                     splitter='best',
                                     random_state=self.random_state)

        # Fit Model
        tree.fit(features, residuals.loc[:, 'residual'])

        return tree

    def _line_search(self, tree, features, residuals, raw_predictions, nn_complexity_measure):
        """
        Step 5 : Line Search Updating terminal regions
        This function calls the loss.update_terminal_region for each leaf
        """
        #Compute leaf values for each sample in dataset, getting the leaf index
        terminal_regions = tree.apply(np.array(features, dtype='float32'))

        #neural net complexity penalty
        complexity_penalty = self.nn_complexity * nn_complexity_measure

        # update each leaf (= perform line search)
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self.loss_._update_terminal_region(tree, terminal_regions, leaf,
                                               raw_predictions, residuals, self.reg_leaf, complexity_penalty)

        terminal_values = tree.value[:, 0, 0].take(terminal_regions, axis=0)

        return terminal_values

    def _update(self, i, terminal_values, raw_predictions):
        """
        Step 6 : Update via Shrinkage
        """
        #Update raw_prediction
        raw_predictions.loc[:,'utility'] += self.learning_rate * terminal_values
        raw_predictions.loc[:,'utility_obs'] = self.loss_._utility_obs(raw_predictions)

        #Save a copy of terminal values
        self.decision_function.loc[:,str(i)] = terminal_values

        return raw_predictions

    def _save_iteration(self, i, nn, tree, col_idx, y, raw_predictions):
        """
        Step 7 : Save components in the iteration
        """
        # Adding NN to ensemble
        self.feature_aug_[i] = nn

        # Saving columns used in nn
        self.col_idx_[i] = col_idx

        # Adding Tree to ensemble
        self.estimators_[i] = tree

        # Updating train_score_
        self.train_score_[i] = self.loss_.loss(y, raw_predictions)

        # Verbose
        if self.verbose:
            self.verbose_reporter.update(i, self)

        return None

    def _post_processing(self, y):
        """
        Step 8 : LASSO Post-Processing - Be aware of singular matrix
        """
        X_post = self.decision_function
        #Create specification dictionary
        model_specification = OrderedDict()
        for variable in X_post.columns[2:]:
            model_specification[variable] = 'all_same'
        zeros = np.zeros(len(model_specification))

        X_post = X_post.merge(y.loc[:,['RARID','HNAME','RESWL']], on=['RARID','HNAME'])
        X_post.reset_index(inplace=True, drop=True)

        self.loss_.init_estimator_warnings('ignore')
        model = self.init_(data=X_post,
                           alt_id_col=self.alt_id_col,
                           obs_id_col=self.obs_id_col,
                           choice_col=self.choice_col,
                           specification=model_specification,
                           model_type="MNL")
        model.fit_mle(zeros, print_res=False, ridge = self.post_lambda)

        self.post_processing_model  = model
        self.summary = self.post_processing_model.get_statsmodels_summary()
        self.loss_.init_estimator_warnings('default')

        # Verbose
        if self.verbose:
            self.verbose_reporter.complete()

        return None

    def predict(self, X):
        """
        Predict class probabilities for X.
        Parameters
        ----------
        X : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, n_features + 2)
            the dataframe should be in long format for the discrete choice model
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        raw_predictions = self._predict_utility_function(X)
        return self.loss_._raw_prediction_to_decision(raw_predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        Parameters
        ----------
        X : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, n_features + 2)
            the dataframe should be in long format for the discrete choice model
        Returns
        -------
        p : Pandas DataFrame with alt_id_col, obs_id_col, proba
            of the shape (n_samples, n_features + 2)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        raw_predictions = self._predict_utility_function(X)
        try :
            return self.loss_._raw_prediction_to_proba(raw_predictions)
        except AttributeError:
            raise AttributeError('loss=%r does not support predict_proba' %
                                 self.loss)

    def _predict_utility_function(self, X):
        """
        Computes the predicted utility function
        Utility Function = Sum of tree predictions + init
        Parameters
        ----------
        X : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, n_features + 2)
        Returns
        -------
        raw_predictions : Pandas DataFrame with alt_id_col, obs_id_col, and utility
            (estimated utility function) of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
        """
        #Validate data
        self._validate_testdata(X)
        raw_predictions = X.loc[:, self.X_base_cols_]
        decisions = X.loc[:, self.X_base_cols_]

        #Get initial prediction
        decisions.loc[:,'init'] = self.dominant_model.params.values * X.loc[:,self.dominant_feature_]

        #Get components
        nns = self.feature_aug_
        trees = self.estimators_
        col_idx = self.col_idx_

        #Predict Stages
        for i in range(self.n_estimators):
            #Pre-Processing
            nn = nns[i]
            col = col_idx[i]
            features = nn.predict(X.loc[:,col].values)

            #Base Learner
            tree = trees[i]
            decisions.loc[:,str(i)] = tree.predict(features)

        #Post-Processing
        self.loss_.init_estimator_warnings('ignore')
        raw_predictions.loc[:,'utility'] = self.post_processing_model.predict(decisions)
        self.loss_.init_estimator_warnings('default')

        return raw_predictions

    def _validate_testdata(self, X):
        """
        Validate Input data : Datatype and existance of alt_id_col and obs_id_col
        Parameters
        ----------
        X : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, n_features + 2)
            the dataframe should be in long format for the discrete choice model
            The input samples.
        """
        # Checks whether X amd y are Pandas DataFrames
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame but "
                            f"was {type(X)}")

        # Checks whether alt_id_col and obs_id_col is in Dataframes X and y
        problem_cols_X = [
            col for col in self.X_base_cols_ if col not in X.columns]

        if problem_cols_X != []:
            raise ValueError("The following columns in are not in "
                             f"X.columns : {problem_cols_X}")

        return None

class Reg_ConditionalLogit_Loss():
    """
    Regularised Conditional Logit Loss Function
    """

    def __init__(self, alt_id_col, obs_id_col, choice_col):
        self.alt_id_col = alt_id_col
        self.obs_id_col = obs_id_col
        self.choice_col = choice_col

    def init_estimator(self):
        """
        Returns
        -------
        Conditional Logistic Regression model class
        """
        return create_choice_model

    def init_estimator_warnings(self, status='ignore'):
        """
        Changes the warnings setting for the init estimator
        """
        warnings.filterwarnings(status, category=FutureWarning)
        warnings.filterwarnings(status, category=UserWarning)
        warnings.filterwarnings(status, category=RuntimeWarning)

        return None

    def loss(self, y, raw_predictions):
        """
        Compute the Log Likelihood Function / Training Score
        Parameters
        ---------
        y : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
            The targets.
        raw_predictions : Pandas DataFrame with alt_id_col, obs_id_col, and utility
            (estimated utility function) of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
        """
        data = y.merge(raw_predictions, on=[self.obs_id_col, self.alt_id_col])

        # Initialise Log Likelihood
        ll = 0
        # Looping over all individuals
        for _, group in data.groupby(self.obs_id_col):
            ll += group.loc[:, 'utility'].dot(group.loc[:, 'RESWL'])
            ll -= logsumexp(group.loc[:, 'utility'])

        return ll

    def _negative_gradient(self, y, raw_predictions):
        """
        Compute negative gradient

        Parameters
        ----------
        y : Pandas DataFrame with alt_id_col and obs_id_col
            of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
            The targets.
        raw_predictions : Pandas DataFrame with alt_id_col, obs_id_col, utility (estimated utility function)
            and utility_obs (sum of all utilities in obs) of the shape (n_samples, 4)
            the dataframe should be in long format for the discrete choice model
        Returns
        -------
        residuals : Pandas DataFrame with alt_id_col, obs_id_col and residual
            of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
        """
        raw_predictions = self._overflow_guard(raw_predictions)

        target = [col for col in y.columns if col not in [self.alt_id_col, self.obs_id_col]][0]

        residuals = y.merge(
            raw_predictions, on=[
                self.alt_id_col, self.obs_id_col])
        residuals.loc[:,'residual'] = residuals.loc[:,target] \
            - np.exp(residuals.loc[:,'utility']) / residuals.loc[:,'utility_obs']

        return residuals.loc[:, [self.alt_id_col, self.obs_id_col, 'residual']]

    def _hessian(self, raw_predictions):
        """
        Compute second derivative of loss function

        Parameters
        ----------
        raw_predictions : Pandas DataFrame with alt_id_col, obs_id_col, utility (estimated utility function)
            and utility_obs (sum of all utilities in obs) of the shape (n_samples, 4)
            the dataframe should be in long format for the discrete choice model
        Returns
        -------
        residuals : Pandas DataFrame with alt_id_col, obs_id_col and residual
            of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
        """
        hessian = self._overflow_guard(raw_predictions)
        hessian.loc[:,'hessian'] = (hessian.loc[:,'utility_obs'] \
                                    - np.exp(hessian.loc[:,'utility']) ** 2) / hessian.loc[:,'utility_obs'] ** 2

        return hessian.loc[:, 'hessian']

    def _update_terminal_region(
            self,
            tree,
            terminal_regions,
            leaf,
            raw_predictions,
            residual,
            reg_leaf,
            complexity_penalty):
        """
        Update the terminal regions (=leaves) of the given tree
        Mutates the tree
        Parameters
        ----------
        tree : tree.Tree
            The tree object
        """
        terminal_region = np.where(terminal_regions == leaf)[0]

        utility_m_1 = raw_predictions.take(terminal_region, axis=0)
        y_true = residual.take(terminal_region, axis=0)

        gradient = self._negative_gradient(y_true, utility_m_1)
        hessian = self._hessian(utility_m_1)

        numerator =  np.sum(gradient.loc[:, 'residual'])  # -1*
        denominator = np.sum(hessian) + reg_leaf

        # Prevents overflow and divison by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = complexity_penalty + \
                (numerator / denominator)

        return None

    def _utility_obs(self, raw_predictions):
        """
        Converts raw prediction of one individual to probabilities summing to 1.
        Parameters
        ---------
        raw_predictions : Pandas DataFrame of one individual with alt_id_col, obs_id_col, and utility
            (estimated utility function) of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
        Returns
        ---------
        utility_sum : Pandas Series of sum of exp(utility) for each obs
            the dataframe should be in long format for the discrete choice model
        """
        raw_predictions_copy = raw_predictions.loc[:, [
            self.obs_id_col, 'utility']].copy()
        raw_predictions_copy = self._overflow_guard(raw_predictions_copy)

        obs = raw_predictions_copy.groupby(self.obs_id_col)
        utility_sum = obs.transform(lambda x: np.sum(np.exp(x)))

        return utility_sum.values

    def _overflow_guard(self, raw_predictions):
        """
        Bound the utility function to avoid exp overflowing
        Parameters
        ---------
        raw_predictions : Pandas DataFrame of one individual with alt_id_col, obs_id_col, and utility
            (estimated utility function) of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
        Returns
        ---------
        raw_predictions : Pandas DataFrame of one individual with alt_id_col, obs_id_col, and utility
            (estimated utility function) of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
        """
        # Define the boundary values which are not to be exceeded during
        max_exp_val = 700
        min_exp_val = -700
        # The following guards against numeric under / over flow in the utility
        # function
        too_large = raw_predictions.loc[:, 'utility'] > max_exp_val
        too_small = raw_predictions.loc[:, 'utility'] < min_exp_val
        raw_predictions.loc[too_large, 'utility'] = max_exp_val
        raw_predictions.loc[too_small, 'utility'] = min_exp_val

        return raw_predictions

    def _raw_prediction_to_proba(self, raw_predictions):
        """
        Converts raw prediction of one individual to probabilities
        summing to 1.
        Parameters
        ---------
        raw_predictions : Pandas DataFrame of one individual with alt_id_col, obs_id_col
            and utility (estimated utility function) of the shape (n_samples, 3)
            the dataframe should be in long format for the discrete choice model
        Returns
        ---------
        probi_predictions : Pandas DataFrame of one individual with alt_id_col, obs_id_col,
            and probability estimates (proba).
            the dataframe should be in long format for the discrete choice model
        """
        raw_predictions_copy = raw_predictions.loc[:, [
            self.alt_id_col, self.obs_id_col, 'utility']].copy()
        raw_predictions_copy = self._overflow_guard(raw_predictions_copy)

        obs = raw_predictions_copy.groupby(self.obs_id_col)
        demom = obs.transform(lambda x: np.sum(np.exp(x)))
        num = obs.transform(np.exp)
        raw_predictions_copy['proba'] = num / demom

        return raw_predictions_copy.loc[:, [self.alt_id_col, self.obs_id_col, 'proba']]

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)

        proba.loc[:,'decision'] = 0
        proba.loc[proba.loc[:,'decision'].idxmax(),'decision'] = 1

        return proba.loc[:, [self.alt_id_col, self.obs_id_col, 'decision']]


class VerboseReporter():
    def __init__(self):
        self.start_time = time.time()
        self.start_next = time.time()

    def start(self):
        self.start_next = time.time()
        print('Starting Training')

    def update(self, iteration, estimator):

        train_score = round(estimator.train_score_[iteration], 2)

        elapsed = time.time() - self.start_time
        remaining = (estimator.n_estimators - iteration) * \
            elapsed / (iteration + 1)
        delta = time.time() - self.start_next

        elapsed = self._format_time(elapsed)
        remaining = self._format_time(remaining)
        delta = self._format_time(delta)

        print(f"{iteration} : train_score : {train_score}     elapsed : {elapsed}"
              f"     delta : {delta}     remaining : {remaining}")
        self.start_next = time.time()

    def _format_time(self, x):
        # Time Formatting
        if x < 60:
            return '{0:.2f}s'.format(x)
        elif x < 3600:
            return '{0:}m'.format(int(x / 60.0)) + ' {0:}s'.format(int(x % 60))
        else:
            return '{0:}h'.format(int(x / 3600.0)) + ' {0:}m'.format(int(x % 60.0)) + ' {0:}s'.format(int(x % 60.0 % 60.0))

    def complete(self):
        elapsed = time.time() - self.start_time
        elapsed = self._format_time(elapsed)
        print(f"The training is completed in {elapsed}.")

