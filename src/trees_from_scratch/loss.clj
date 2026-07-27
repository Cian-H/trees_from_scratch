(ns trees-from-scratch.loss
  (:require [trees-from-scratch.maths :refer [log2 mean median]]
            [trees-from-scratch.utils :refer [class-probabilities]]))

(defn gini [labels]
  (if (empty? labels)
    0.0
    (->> (class-probabilities labels)
         (map #(* % %))
         (reduce + 0.0)
         (- 1.0))))

(defn entropy [labels]
  (->> (class-probabilities labels)
       (filter pos?)
       (map (fn [p] (* p (log2 p))))
       (reduce + 0.0)
       -))

(defn misclassification-rate [labels]
  (let [probs (class-probabilities labels)]
    (if (empty? probs)
      0.0
      (- 1.0 (reduce max probs)))))

(defn mean-squared-deviation [values]
  (if (empty? values)
    0.0
    (let [avg (mean values)]
      (/ (->> values
              (map #(- avg %))
              (map #(* % %))
              (reduce + 0.0))
         (count values)))))

(defn mean-absolute-deviation [values]
  (if (empty? values)
    0.0
    (let [med (median values)]
      (/ (->> values
              (map #(- med %))
              (map abs)
              (reduce + 0.0))
         (count values)))))

(defn poisson-deviance [values]
  (if (empty? values)
    0.0
    (let [avg (mean values)]
      (->> values
           (filter pos?)
           (map #(- (* % (Math/log (/ % avg)))
                    (- % avg)))
           (reduce + 0.0)
           (* 2.0)))))
