(ns random-forest-regression
  (:require [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.models.cart :as cart]
            [trees-from-scratch.models.random-forest :as rf]
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

(println "\n--- CART ---")
(println "Fitting CART tree (Regression)...")
(def cart-tree (cart/train train-dataset target {:task-type :regression}))
(def cart-train-r2 (evaluation/evaluate cart-tree train-dataset target :r2))
(def cart-test-r2 (evaluation/evaluate cart-tree test-dataset target :r2))
(println (format "CART Training R^2: %.4f" cart-train-r2))
(println (format "CART Testing R^2:  %.4f" cart-test-r2))
(println (format "CART Tree Depth: %d" (tree-core/tree-depth cart-tree)))

(println "\n--- Random Forest ---")
(println "Fitting Random Forest (Regression)...")
(def rf-model (rf/train train-dataset target {:task-type :regression :max-trees 50 :window-size 50}))
(def rf-train-r2 (evaluation/evaluate rf-model train-dataset target :r2))
(def rf-test-r2 (evaluation/evaluate rf-model test-dataset target :r2))
(println (format "RF Training R^2: %.4f" rf-train-r2))
(println (format "RF Testing R^2:  %.4f" rf-test-r2))
(println (format "RF Number of Trees: %d" (count (tree-core/get-trees rf-model))))
(println (format "RF Max Tree Depth: %d" (tree-core/tree-depth rf-model)))
