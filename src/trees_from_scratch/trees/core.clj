(ns trees-from-scratch.trees.core
  "Universal interface for various kinds of decision and regression trees.")

(defprotocol Tree
  (predict [this row] "Predicts the target value for a single row.")
  (tree-depth [this] "Calculates the maximum depth of the given tree.")
  (display-tree
    [this]
    [this depth branch-label]
    "Pretty-prints the tree to the console.")
  (get-trees [this] "Returns all trees that comprise the model"))

(extend-protocol Tree
  nil
  (predict [_ _] nil)
  (tree-depth [_] 0)
  (display-tree
    ([_])
    ([_ _ _]))
  (get-trees [_]))
