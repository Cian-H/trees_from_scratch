(ns trees-from-scratch.spatial.hyperplane
  "Shared logic for hyperplane-based tree models (Oblique, Random Projection)."
  (:require [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.models.core :as core]))

;; -- Categorical splits remain the same (axis-aligned) --
(defn vector-splits-categorical [v] (distinct v))

(defn partition-labels-categorical [v labels split-val]
  (reduce (fn [[l r] [val label]]
            (if (= val split-val)
              [(conj l label) r]
              [l (conj r label)]))
          [[] []]
          (map vector v labels)))

(defn best-categorical-split [dataset feat target-key loss-fn]
  (let [v (ds/get-column dataset feat)
        parent-labels (ds/get-column dataset target-key)]
    (when-let [best-split-val
               (->> (vector-splits-categorical v)
                    (map (fn [split-val]
                           (let [[left-labels right-labels] (partition-labels-categorical v parent-labels split-val)]
                             {:split-val split-val
                              :loss-reduction (core/loss-reduction parent-labels left-labels right-labels loss-fn)})))
                    (reduce (fn [best current]
                              (if (and (> (:loss-reduction current) 0.0)
                                       (or (nil? best) (> (:loss-reduction current) (:loss-reduction best))))
                                current
                                best))
                            nil))]
      (let [[left-ds right-ds] (ds/split-by-categorical dataset feat (:split-val best-split-val))]
        (assoc best-split-val
               :type :categorical
               :feature feat
               :left left-ds
               :right right-ds)))))

(defn generate-candidate-hyperplanes
  "Generates M random candidate hyperplanes. Each candidate is {:weights W :threshold T}.
   Thresholds are chosen by projecting a random row."
  [continuous-features dataset M ^java.util.Random rng]
  (let [num-rows (ds/row-count dataset)]
    (if (or (zero? num-rows) (empty? continuous-features))
      []
      (vec (for [_ (range M)]
             (let [weights (into {} (map (fn [f] [f (- (* 2.0 (.nextDouble rng)) 1.0)]) continuous-features))
                   rand-idx (.nextInt rng num-rows)
                   ;; Pick a threshold by projecting a random row
                   threshold (reduce-kv (fn [acc k w]
                                          (+ acc (* (double w) (double (nth (ds/get-column dataset k) rand-idx 0.0)))))
                                        0.0
                                        weights)]
               {:weights weights :threshold threshold}))))))

(defn evaluate-hyperplane
  "Evaluates a candidate hyperplane on a subset of indices."
  [dataset indices candidate target-key loss-fn]
  (let [{:keys [weights threshold]} candidate
        parent-labels (ds/get-column dataset target-key)
        subset-labels (mapv #(nth parent-labels %) indices)
        [left-labels right-labels]
        (reduce (fn [[l r] idx]
                  (let [projected-val (reduce-kv (fn [acc k w]
                                                   (+ acc (* (double w) (double (nth (ds/get-column dataset k) idx 0.0)))))
                                                 0.0
                                                 weights)]
                    (if (<= projected-val threshold)
                      [(conj l (nth parent-labels idx)) r]
                      [l (conj r (nth parent-labels idx))])))
                [[] []]
                indices)]
    (assoc candidate :loss-reduction (core/loss-reduction subset-labels left-labels right-labels loss-fn))))
