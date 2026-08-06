(ns cart-regression
  (:require [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.models.cart :as cart]
            [trees-from-scratch.metrics.evaluation :as evaluation]
            [trees-from-scratch.trees.core :as tree-core]
            [utils]))

(def dataset (utils/fetch-tips))
(def total-rows (ds/row-count dataset))
(println (format "Loaded dataset with %d rows." total-rows))

(def split (utils/train-test-split dataset 0.8))
(def train-dataset (:train split))
(def test-dataset (:test split))

(println (format "Split dataset: %d train rows, %d test rows." (ds/row-count train-dataset) (ds/row-count test-dataset)))

(def target :tip)

(println "Fitting CART tree (Regression)...")
(def tree (cart/train train-dataset target {:task-type :regression}))

(println "Tree fitted successfully.")
(println "Evaluating fit on training data...")
(def train-r2 (evaluation/evaluate tree train-dataset target :r2))
(println (format "Training R^2: %.4f" train-r2))

(println "Evaluating fit on testing data...")
(def test-r2 (evaluation/evaluate tree test-dataset target :r2))
(println (format "Testing R^2: %.4f" test-r2))
(tree-core/display-tree tree)
