(ns trees-from-scratch.models.core
  "Core model evaluation and prediction functions."
  (:require [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.trees.core :as tree-core]))

(defn majority-class
  "Finds the most frequent class in a sequence of labels."
  [labels]
  (when (seq labels)
    (->> labels
         frequencies
         (sort-by val >)
         ffirst)))

(defn mean-target
  "Calculates the mean value of a sequence of continuous targets."
  [values]
  (when values
    (if (empty? values)
      0.0
      (/ (apply + values) (count values)))))

(defn predict-all
  "Predicts the target for all rows in a dataset using the given tree."
  [tree dataset]
  (map (fn [idx]
         (tree-core/predict tree (ds/get-row dataset idx)))
       (range (ds/row-count dataset))))

(defn evaluate
  "Evaluates a tree on a dataset against the specified metric (e.g., :accuracy, :mse)."
  ([tree dataset target-key]
   (evaluate tree dataset target-key :accuracy))
  ([tree dataset target-key metric]
   (let [predictions (predict-all tree dataset)
         actuals     (ds/get-column dataset target-key)]
     (if (empty? actuals)
       0.0
       (case metric
         :accuracy (let [correct (reduce + 0 (map #(if (= %1 %2) 1 0) predictions actuals))]
                     (/ (double correct) (count actuals)))
         :mse      (let [errors (map (fn [p a] (let [d (- p a)] (* d d))) predictions actuals)]
                     (/ (reduce + 0.0 errors) (count actuals)))
         :r2       (let [mean-actual (mean-target actuals)
                         sst         (reduce + 0.0 (map (fn [a] (let [d (- a mean-actual)] (* d d))) actuals))
                         sse         (reduce + 0.0 (map (fn [p a] (let [d (- p a)] (* d d))) predictions actuals))]
                     (if (zero? sst)
                       0.0
                       (- 1.0 (/ sse sst)))))))))
