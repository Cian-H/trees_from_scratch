(ns trees-from-scratch.models.core
  (:require [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.trees.binary :as binary-tree]))

(defn majority-class
  [labels]
  (when (seq labels)
    (->> labels
         frequencies
         (sort-by val >)
         ffirst)))

(defn mean-target
  [values]
  (when values
    (if (empty? values)
      0.0
      (/ (apply + values) (count values)))))

(defn predict-all
  [tree dataset]
  (map (fn [idx]
         (binary-tree/predict tree (ds/get-row dataset idx)))
       (range (ds/row-count dataset))))

(defn evaluate
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
                     (/ (reduce + 0.0 errors) (count actuals))))))))
