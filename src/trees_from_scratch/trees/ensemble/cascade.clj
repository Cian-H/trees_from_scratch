(ns trees-from-scratch.trees.ensemble.cascade
  "Cascade (sequential) tree ensemble representation and evaluation."
  (:require [trees-from-scratch.trees.core :as tree-core]))

(defn default-cascade-fn
  "Injects all previous predictions into the data row using the tree index as the key,
   as well as setting :prev-pred to the latest prediction."
  [original-row preds-map]
  (let [latest-idx (dec (count preds-map))
        latest-pred (get preds-map latest-idx)]
    (reduce-kv (fn [row tree-idx pred]
                 (assoc row (keyword (str "tree-" tree-idx)) pred))
               (assoc original-row :prev-pred latest-pred)
               preds-map)))

(defn markov-cascade [original-row preds-map]
  (let [latest-idx (dec (count preds-map))]
    (assoc original-row :prev-pred (get preds-map latest-idx))))

(defrecord CascadeEnsemble [agg cascade-fn trees]
  tree-core/Tree

  (predict [_ data-row]
    (let [preds (loop [remaining trees
                       current-row data-row
                       acc []
                       tree-idx 0]
                  (if-let [tree (first remaining)]
                    (let [pred (tree-core/predict tree current-row)
                          next-preds (conj acc pred)
                          preds-map (zipmap (range (inc tree-idx)) next-preds)
                          next-row (cascade-fn data-row preds-map)]
                      (recur (rest remaining) next-row next-preds (inc tree-idx)))
                    acc))]
      (if agg
        (agg preds)
        (last preds))))

  (tree-depth [_]
    (if (seq trees)
      (reduce + (map tree-core/tree-depth trees))
      0))

  (display-tree [this]
    (println "Cascade Ensemble Tree:")
    (tree-core/display-tree this 0 "Root: "))

  (display-tree [_ depth branch-label]
    (let [indent (apply str (repeat (* 2 depth) " "))]
      (println indent branch-label "Cascade Ensemble of" (count trees) "trees")))

  (get-trees [_] trees))

(defn make
  "Creates a CascadeEnsemble instance. If the first argument after `agg` is a function, uses it as `cascade-fn`; otherwise uses `default-cascade-fn`."
  [agg & args]
  (let [[casc-fn trees] (if (fn? (first args))
                          [(first args) (rest args)]
                          [default-cascade-fn args])]
    (->CascadeEnsemble agg casc-fn trees)))
