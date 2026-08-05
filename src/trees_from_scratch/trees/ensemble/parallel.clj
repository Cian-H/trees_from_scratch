(ns trees-from-scratch.trees.ensemble.parallel
  "Parallel tree ensemble representation and evaluation."
  (:require [trees-from-scratch.trees.core :as tree-core]))

(defrecord ParallelEnsemble [agg trees]
  tree-core/Tree
  (predict [_ data-row]
    (let [preds (map #(tree-core/predict % data-row) trees)]
      (agg preds)))
  (tree-depth [_]
    (if (seq trees)
      (apply max (map tree-core/tree-depth trees))
      0))
  (display-tree [this]
    (println "Parallel Ensemble Tree:")
    (tree-core/display-tree this 0 "Root: "))
  (display-tree [_ depth branch-label]
    (let [indent (apply str (repeat (* 2 depth) " "))]
      (println indent branch-label "Parallel Ensemble of" (count trees) "trees")))
  (get-trees [_] trees))

;; Alias for backward compatibility if referenced directly
(def Ensemble ParallelEnsemble)
(def ->Ensemble ->ParallelEnsemble)

(defn make
  "Creates a ParallelEnsemble instance."
  [agg & trees]
  (->ParallelEnsemble agg trees))
