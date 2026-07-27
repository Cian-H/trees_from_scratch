(ns trees-from-scratch.tree)

(defn make-leaf [prediction]
  {:prediction prediction})

(defn make-node [rule left right]
  {:rule rule :left left :right right})

(defn tree-depth [{:keys [left right] :as tree}]
  (if tree
    (inc (max (tree-depth left) (tree-depth right)))
    0))

(defn predict [{:keys [prediction rule left right]} data-row]
  (if prediction
    prediction
    (let [{:keys [feature threshold]} rule]
      (if (<= (get data-row feature) threshold)
        (recur left data-row)
        (recur right data-row)))))

(defn display
  "Pretty-prints the decision tree to the console."
  ([tree]
   (println "Decision Tree:")
   (display tree 0 "Root: "))

  ([{:keys [prediction rule left right]} depth branch-label]
   (let [indent (apply str (repeat (* 2 depth) " "))] ; 2 spaces per depth level
     (if prediction
       (println indent branch-label "Predict ->" prediction)
       (let [{:keys [feature threshold]} rule]
         (println indent branch-label "Is" feature "<=" threshold "?")
         (display left (inc depth) "├── True:  ")
         (display right (inc depth) "└── False: "))))))
