(ns trees-from-scratch.trees.binary
  "Binary decision tree representation and evaluation."
  (:require [trees-from-scratch.trees.core :as tree-core]))

(defrecord BinaryLeaf [prediction]
  tree-core/Tree
  (predict [_ _] prediction)
  (tree-depth [_] 1)
  (display-tree [this]
    (println "Decision Tree:")
    (tree-core/display-tree this 0 "Root: "))
  (display-tree [_ depth branch-label]
    (let [indent (apply str (repeat (* 2 depth) " "))]
      (println indent branch-label "Predict ->" prediction)))
  (get-trees [this] [this]))

(defn make-leaf
  "Creates a leaf node for a decision tree, containing the final prediction."
  [prediction]
  (->BinaryLeaf prediction))

(defrecord BinarySplit [description predicate left right]
  tree-core/Tree
  (predict [_ data-row]
    (let [go-left? (predicate data-row)
          next-node (if go-left? left right)]
      (if next-node
        (tree-core/predict next-node data-row)
        nil)))
  (tree-depth [_]
    (inc (max (if left (tree-core/tree-depth left) 0)
              (if right (tree-core/tree-depth right) 0))))
  (display-tree [this]
    (println "Decision Tree:")
    (tree-core/display-tree this 0 "Root: "))
  (display-tree [_ depth branch-label]
    (let [indent (apply str (repeat (* 2 depth) " "))]
      (println indent branch-label description)
      (when left (tree-core/display-tree left (inc depth) "├── True:  "))
      (when right (tree-core/display-tree right (inc depth) "└── False: "))))
  (get-trees [this] [this]))

(defn make-split
  "Creates a generic split (decision) node, dispatching data based on a predicate."
  [description predicate left right]
  (->BinarySplit description predicate left right))

(defn make-continuous-split
  "Creates a continuous split node that evaluates whether a feature is <= threshold."
  [feature threshold left right]
  (let [desc (str "Is " feature " <= " threshold " ?")
        pred (fn [row]
               (let [v (get row feature)]
                 (and (some? v) (<= v threshold))))]
    (make-split desc pred left right)))

(defn make-categorical-split
  "Creates a categorical split node that evaluates whether a feature == category."
  [feature category left right]
  (let [desc (str "Is " feature " == " category " ?")
        pred (fn [row] (= (get row feature) category))]
    (make-split desc pred left right)))
