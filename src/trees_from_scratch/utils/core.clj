(ns trees-from-scratch.utils.core
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

(defn seeded-shuffle
  "Shuffles a collection using the given java.util.Random instance."
  [^java.util.Random rng coll]
  (let [al (java.util.ArrayList. ^java.util.Collection coll)]
    (java.util.Collections/shuffle al rng)
    (vec al)))

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
