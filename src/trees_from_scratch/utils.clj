(ns trees-from-scratch.utils)

(defn class-probabilities [labels]
  (let [n (count labels)]
    (if (zero? n)
      []
      (->> labels
           frequencies
           vals
           (map #(/ (double %) n))))))
