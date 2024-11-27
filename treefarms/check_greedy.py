# get subset of rashomon set that's greedy past a certain depth
import pandas as pd
import numpy as np
from gosdt._tree import Node, Leaf
import argparse
from sklearn.model_selection import train_test_split
import time
import json

def entropy(ps):
    """
    Calculate the entropy of a given list of binary labels.
    ps[0] looks like it's the proportion of positive labels, 
    ps[1] is proportion of negative ones.
    """
    p_positive = ps[0]
    if p_positive == 0 or p_positive == 1:
        return 0  # Entropy is 0 if all labels are the same
    entropy_val = - (p_positive * np.log2(p_positive) +
                        (1 - p_positive) * np.log2(1 - p_positive))
    return entropy_val

def find_best_feature_to_split_on(X_train,y_train):
        num_features = X_train.shape[1]
        max_gain = -10
        gain_of_feature_to_split = 0
        p_original = np.mean(y_train) # mean of all training labels (aka percentage that are pos)
        entropy_original = entropy([p_original, 1-p_original]) # get entropy of all data (basically treating it like a tree with 1 leaf)
        best_feature = -1
        for feature in range(num_features):
            # Left child labels
            p_left = np.mean(y_train[X_train.iloc[:, feature] == 1])
            
            # Right child labels
            p_right = np.mean(y_train[X_train.iloc[:, feature] == 0])

            p_left = 0 if np.isnan(p_left) else p_left
            p_right = 0 if np.isnan(p_right) else p_right
        
            entropy_left = entropy(np.array([p_left, 1 - p_left]))
            
            entropy_right = entropy(np.array([p_right, 1 - p_right]))
            
            proportion_of_examples_in_left_leaf = (np.sum(X_train.iloc[:, feature] == 1) / len(X_train))
            proportion_of_examples_in_right_leaf = (np.sum(X_train.iloc[:, feature] == 0) / len(X_train))
            gain = entropy_original - ( proportion_of_examples_in_left_leaf* entropy_left +
                                        proportion_of_examples_in_right_leaf* entropy_right)
            if gain >= max_gain:
                max_gain = gain
                best_feature = feature

        return best_feature

def go_to_depth(tree_json, depth):
    # tree is json version of tree classifier object

    if type(tree_json) != dict:
        tree_json = json.loads(tree_json.json())

    # base case, tree is just a leaf
    if "true" not in tree_json:
        return 
    # correct depth reached
    if depth == 1:
        return [tree_json]
    if depth == 2:
        return [tree_json["true"], tree_json["false"]]
    # recursion, keep going
    else: 
        left_tree = go_to_depth(tree_json["true"], depth-1)
        right_tree = go_to_depth(tree_json["false"], depth-1)
        if not left_tree:
            if right_tree:
                return right_tree
            return
        if not right_tree:
            if left_tree:
                return left_tree
            return
        else:
            return left_tree + right_tree

def is_greedy(tree, X_train, y_train):
    # check if first split of tree is greedy or not.
    # if tree is empty?

    # if tree is a leaf, all leaves are considered greedy
    if "prediction" in tree:
        return True
    # else check if it split on the greediest feature
    else:
        best_feature = find_best_feature_to_split_on(X_train, y_train)
        if tree["feature"] == best_feature:
            return True
        return False

def check_greedy(tree, X_train, y_train, depth):
    # tree is tree classifier object
    # X_train, y_train are data and labels
    # depth is how deep to start checking

    # step 0. turn tree classifier into json
    tree_json = json.loads(tree.json())
    # step 1. go to specified depth of tree, get left and right trees
    subtrees = go_to_depth(tree_json, depth)
    if len(subtrees) > 1:
        left_tree = subtrees[0]
        right_tree = subtrees[1]
    else: # checking whole tree - TODO: not fully implemented this below
        left_tree = subtrees[0]
        right_tree = None
    # step 2. check if left and right trees are greedy

    # what if the left or right trees end up empty? 
    # for example if there's no left tree anymore at depth 3
    # do i return none? or would that mess up the result
    # maybe i just return True and let the function keep checking the other branch?

    # left or right tree aren't leaves, check if the next split is greedy
    if is_greedy(left_tree) == False:
        return False
    if is_greedy(right_tree) == False:
        return False
    else:
        # go down the tree
        # TODO: have to get X_train and y_train that correspond to the split made
        left_split_feature = left_tree["feature"]
        right_split_feature = right_tree["feature"]
        X_train_left = X_train[X_train.iloc[:, left_split_feature]]
        y_train_left = y_train[y_train.iloc[:, left_split_feature]]
        X_train_right = X_train[X_train.iloc[:, right_split_feature]]
        y_train_right = y_train[y_train.iloc[:, right_split_feature]]
        left_greedy = check_greedy(left_tree, X_train_left, y_train_left, depth=depth+1) # is this correct?
        right_greedy = check_greedy(right_tree, X_train_left, y_train_left, depth=depth+1)
        return (left_greedy and right_greedy)