(ns trees-from-scratch.trees.ensemble
  "Tree ensemble representation and evaluation."
  (:require [trees-from-scratch.trees.core :as tree-core]))

(defrecord Ensemble [agg trees]
  tree-core/Tree
  (predict [_ data-row]
    (let [preds (map #(tree-core/predict % data-row) trees)]
      (agg preds)))
  (tree-depth [_]
    (if (seq trees)
      (apply max (map tree-core/tree-depth trees))
      0))
  (display-tree [this]
    (println "Ensemble Tree:")
    (tree-core/display-tree this 0 "Root: "))
  (display-tree [_ depth branch-label]
    (let [indent (apply str (repeat (* 2 depth) " "))]
      (println indent branch-label "Ensemble of" (count trees) "trees"))))

(defn make [agg & trees]
  (->Ensemble agg trees))
