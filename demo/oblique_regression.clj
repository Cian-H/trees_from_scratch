(ns oblique-regression
  (:require [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.models.cart :as cart]
            [trees-from-scratch.models.oblique :as obl]
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

(println "\n--- Standard CART ---")
(println "Fitting CART tree (Regression)...")
(def cart-tree (cart/train train-dataset target {:task-type :regression}))
(def cart-train-r2 (evaluation/evaluate cart-tree train-dataset target :r2))
(def cart-test-r2 (evaluation/evaluate cart-tree test-dataset target :r2))
(println (format "CART Training R^2: %.4f" cart-train-r2))
(println (format "CART Testing R^2:  %.4f" cart-test-r2))
(println (format "CART Tree Depth: %d" (tree-core/tree-depth cart-tree)))

(println "\n--- Oblique CART ---")
(println "Fitting Oblique tree (Regression)...")
(def obl-tree (obl/train train-dataset target {:task-type :regression :num-candidates 100}))
(def obl-train-r2 (evaluation/evaluate obl-tree train-dataset target :r2))
(def obl-test-r2 (evaluation/evaluate obl-tree test-dataset target :r2))
(println (format "Oblique Training R^2: %.4f" obl-train-r2))
(println (format "Oblique Testing R^2:  %.4f" obl-test-r2))
(println (format "Oblique Tree Depth: %d" (tree-core/tree-depth obl-tree)))
