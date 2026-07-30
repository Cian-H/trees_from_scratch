(ns trees-from-scratch.trees.binary
  "Binary decision tree representation and evaluation.")

(defn make-leaf
  "Creates a leaf node for a decision tree, containing the final prediction."
  [prediction]
  {:type       :leaf
   :prediction prediction})

(defn make-split
  "Creates a generic split (decision) node, dispatching data based on a predicate."
  [description predicate left right]
  {:type        :split
   :description description
   :predicate   predicate
   :left        left
   :right       right})

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

(defn predict
  "Traverses the tree to make a prediction for a single data row."
  [node data-row]
  (loop [curr node]
    (when curr
      (case (:type curr)
        :leaf  (:prediction curr)
        :split (let [go-left?   ((:predicate curr) data-row)
                     next-node (if go-left? (:left curr) (:right curr))]
                 (recur next-node))
        nil))))

(defn tree-depth
  "Calculates the maximum depth of the given tree node (leaves have depth 1)."
  [node]
  (if node
    (case (:type node)
      :leaf  1
      :split (inc (max (tree-depth (:left node))
                       (tree-depth (:right node)))))
    0))

(defn display
  "Pretty-prints the decision tree to the console."
  ([tree]
   (println "Decision Tree:")
   (display tree 0 "Root: "))

  ([node depth branch-label]
   (let [indent (apply str (repeat (* 2 depth) " "))]
     (case (:type node)
       :leaf  (println indent branch-label "Predict ->" (:prediction node))
       :split (do
                (println indent branch-label (:description node))
                (display (:left node) (inc depth) "├── True:  ")
                (display (:right node) (inc depth) "└── False: "))))))
