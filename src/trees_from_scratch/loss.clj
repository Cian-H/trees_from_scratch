(ns trees-from-scratch.loss
  "Loss functions and impurity metrics for classification and regression trees."
  (:require [trees-from-scratch.maths :refer [log2 mean median]]
            [trees-from-scratch.utils :refer [class-probabilities]]))

(defn gini
  "Calculates the Gini impurity for a sequence of categorical labels."
  [labels]
  (if (empty? labels)
    0.0
    (->> (class-probabilities labels)
         (map #(* % %))
         (reduce + 0.0)
         (- 1.0))))

(defn entropy
  "Calculates the Shannon entropy for a sequence of categorical labels."
  [labels]
  (if (empty? labels)
    0.0
    (->> (class-probabilities labels)
         (filter pos?)
         (map (fn [p] (* p (log2 p))))
         (reduce + 0.0)
         -)))

(defn misclassification-rate
  "Calculates the misclassification rate (1 - max class probability) for a sequence of labels."
  [labels]
  (let [probs (class-probabilities labels)]
    (if (empty? probs)
      0.0
      (- 1.0 (reduce max probs)))))

(defn mean-squared-deviation
  "Calculates the mean squared deviation (variance) for a sequence of continuous values."
  [values]
  (if (empty? values)
    0.0
    (let [avg (mean values)]
      (/ (->> values
              (map #(- avg %))
              (map #(* % %))
              (reduce + 0.0))
         (count values)))))

(defn mean-absolute-deviation
  "Calculates the mean absolute deviation from the median for a sequence of continuous values."
  [values]
  (if (empty? values)
    0.0
    (let [med (median values)]
      (/ (->> values
              (map #(- med %))
              (map abs)
              (reduce + 0.0))
         (count values)))))

(defn poisson-deviance
  "Calculates the Poisson deviance for a sequence of counts or frequencies."
  [values]
  (if (empty? values)
    0.0
    (let [avg (mean values)]
      (if (<= avg 0.0)
        0.0
        (->> values
             (filter pos?)
             (map #(- (* % (Math/log (/ % avg)))
                      (- % avg)))
             (reduce + 0.0)
             (* 2.0))))))

