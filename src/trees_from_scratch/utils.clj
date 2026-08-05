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

(defn rename-key [m old-key new-key]
  (if (contains? m old-key)
    (-> m
        (assoc new-key (get m old-key))
        (dissoc old-key))
    m))
