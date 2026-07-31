(ns cart-classifier
  (:require [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.models.cart :as cart]
            [trees-from-scratch.models.core :as models.core]
            [trees-from-scratch.trees.binary :as btree]
            [utils]))

(def dataset (utils/fetch-iris))
(def total-rows (ds/row-count dataset))
(println (format "Loaded dataset with %d rows." total-rows))

(def split (utils/train-test-split dataset 0.8))
(def train-dataset (:train split))
(def test-dataset (:test split))

(println (format "Split dataset: %d train rows, %d test rows." (ds/row-count train-dataset) (ds/row-count test-dataset)))

(def target :species)

(println "Fitting CART tree...")
(def tree (cart/train train-dataset target {:task-type :classification}))

(println "Tree fitted successfully.")
(println "Evaluating fit on training data...")
(def train-accuracy (models.core/evaluate tree train-dataset target :accuracy))
(println (format "Training Accuracy: %.2f%%" (* 100.0 train-accuracy)))

(println "Evaluating fit on testing data...")
(def test-accuracy (models.core/evaluate tree test-dataset target :accuracy))
(println (format "Testing Accuracy: %.2f%%" (* 100.0 test-accuracy)))
(btree/display tree)
