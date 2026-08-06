(ns trees-from-scratch.models.random-projection
  "Implementation of Random Projection Trees for continuous features."
  (:require [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.trees.stopping :as stopping]
            [trees-from-scratch.metrics.loss :as loss]
            [trees-from-scratch.metrics.evaluation :as evaluation]
            [trees-from-scratch.utils.core :as utils]
            [trees-from-scratch.trees.builder :as builder]))

(defn random-oblique-split
  "Generates a single completely random projection vector and picks a threshold uniformly between min and max."
  [dataset continuous-features target-key ^java.util.Random rng]
  (let [num-rows (ds/row-count dataset)]
    (if (or (zero? num-rows) (empty? continuous-features))
      nil
      (let [weights (into {} (map (fn [f] [f (- (* 2.0 (.nextDouble rng)) 1.0)]) continuous-features))
            ;; Project all rows to find min and max
            projected-vals (map (fn [idx]
                                  (reduce-kv (fn [acc k w]
                                               (+ acc (* (double w) (double (nth (ds/get-column dataset k) idx 0.0)))))
                                             0.0
                                             weights))
                                (range num-rows))
            min-val (apply min projected-vals)
            max-val (apply max projected-vals)]
        (if (= min-val max-val)
          nil ;; Can't split if all projected values are the same
          (let [threshold (+ min-val (* (.nextDouble rng) (- max-val min-val)))
                candidate {:weights weights :threshold threshold}
                ;; Evaluate it just to get the loss reduction and left/right labels
                parent-labels (ds/get-column dataset target-key)
                [left-labels right-labels]
                (reduce (fn [[l r] [p-val label]]
                          (if (<= p-val threshold)
                            [(conj l label) r]
                            [l (conj r label)]))
                        [[] []]
                        (map vector projected-vals parent-labels))
                loss-reduction-val (evaluation/loss-reduction parent-labels left-labels right-labels loss/mean-squared-deviation)
                [left-ds right-ds] (ds/split-by-oblique dataset weights threshold)]

            (assoc candidate
                   :type :oblique
                   :loss-reduction loss-reduction-val
                   :left left-ds
                   :right right-ds)))))))

(defn train
  "Trains a Random Projection Tree on the given dataset."
  ([dataset target-key]
   (train dataset target-key {}))

  ([dataset target-key {:keys [task-type stop]
                        :or   {task-type :regression
                               stop      stopping/cart-stopping-strategy}
                        :as   opt}]
   (if (not= task-type :regression)
     (throw (ex-info "Random Projection trees currently only support regression" {:task-type task-type}))
     (let [rng         (java.util.Random. (hash dataset))
           loss-fn     (builder/default-loss-fn task-type)
           best-split-fn (fn [ds tgt o]
                           (let [feats (or (:features o) (remove #{tgt} (ds/column-names ds)))
                                 ;; Only allow continuous features for Random Projection trees
                                 cont-features (filter #(= (ds/get-type ds %) :continuous) feats)
                                 m (:max-features o)
                                 sampled (if m (vec (take m (utils/seeded-shuffle rng cont-features))) cont-features)]
                             (random-oblique-split ds sampled tgt rng)))]
       (builder/build-tree dataset target-key (assoc opt :loss-fn loss-fn :stop stop :task-type task-type) best-split-fn)))))
