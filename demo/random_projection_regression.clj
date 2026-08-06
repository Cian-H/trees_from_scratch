(ns random-projection-regression
  (:require [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.models.cart :as cart]
            [trees-from-scratch.models.random-projection :as rp]
            [trees-from-scratch.models.core :as models.core]
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
(def cart-train-r2 (models.core/evaluate cart-tree train-dataset target :r2))
(def cart-test-r2 (models.core/evaluate cart-tree test-dataset target :r2))
(println (format "CART Training R^2: %.4f" cart-train-r2))
(println (format "CART Testing R^2:  %.4f" cart-test-r2))
(println (format "CART Tree Depth: %d" (tree-core/tree-depth cart-tree)))

(println "\n--- Random Projection Tree ---")
(println "Fitting Random Projection tree (Regression)...")
(def rp-tree (rp/train train-dataset target {:task-type :regression}))
(def rp-train-r2 (models.core/evaluate rp-tree train-dataset target :r2))
(def rp-test-r2 (models.core/evaluate rp-tree test-dataset target :r2))
(println (format "RP Training R^2: %.4f" rp-train-r2))
(println (format "RP Testing R^2:  %.4f" rp-test-r2))
(println (format "RP Tree Depth: %d" (tree-core/tree-depth rp-tree)))
