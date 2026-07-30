(ns trees-from-scratch.utils
  "General utility functions.")

(defn class-probabilities
  "Calculates the probabilities of each unique class label in the given sequence."
  [labels]
  (let [n (count labels)]
    (if (zero? n)
      []
      (->> labels
           frequencies
           vals
           (map #(/ (double %) n))))))
