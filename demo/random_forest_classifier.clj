(ns random-forest-classifier
  (:require [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.models.cart :as cart]
            [trees-from-scratch.models.random-forest :as rf]
            [trees-from-scratch.metrics.evaluation :as evaluation]
            [trees-from-scratch.trees.core :as tree-core]
            [utils]))

(def dataset (utils/fetch-iris))
(def total-rows (ds/row-count dataset))
(println (format "Loaded dataset with %d rows." total-rows))

(def split (utils/train-test-split dataset 0.8))
(def train-dataset (:train split))
(def test-dataset (:test split))

(println (format "Split dataset: %d train rows, %d test rows." (ds/row-count train-dataset) (ds/row-count test-dataset)))

(def target :species)

(println "\n--- CART ---")
(println "Fitting CART tree...")
(def cart-tree (cart/train train-dataset target {:task-type :classification}))
(def cart-train-acc (evaluation/evaluate cart-tree train-dataset target :accuracy))
(def cart-test-acc (evaluation/evaluate cart-tree test-dataset target :accuracy))
(println (format "CART Training Accuracy: %.2f%%" (* 100.0 cart-train-acc)))
(println (format "CART Testing Accuracy: %.2f%%" (* 100.0 cart-test-acc)))
(println (format "CART Tree Depth: %d" (tree-core/tree-depth cart-tree)))

(println "\n--- Random Forest ---")
(println "Fitting Random Forest...")
(def rf-model (rf/train train-dataset target {:task-type :classification :max-trees 50 :window-size 50}))
(def rf-train-acc (evaluation/evaluate rf-model train-dataset target :accuracy))
(def rf-test-acc (evaluation/evaluate rf-model test-dataset target :accuracy))
(println (format "RF Training Accuracy: %.2f%%" (* 100.0 rf-train-acc)))
(println (format "RF Testing Accuracy: %.2f%%" (* 100.0 rf-test-acc)))
(println (format "RF Number of Trees: %d" (count (tree-core/get-trees rf-model))))
(println (format "RF Max Tree Depth: %d" (tree-core/tree-depth rf-model)))
