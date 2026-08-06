(ns trees-from-scratch.models.oblique
  "Implementation of Oblique Regression Trees using Successive Halving for split optimization."
  (:require [trees-from-scratch.metrics.loss :as loss]
            [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.trees.stopping :as stopping]
            [trees-from-scratch.utils.core :as utils]
            [trees-from-scratch.spatial.hyperplane :as hp]
            [trees-from-scratch.trees.builder :as builder]))

(defn successive-halving-search
  "Performs successive halving to find the best candidate hyperplane."
  [dataset candidates target-key loss-fn ^java.util.Random rng]
  (let [num-rows (ds/row-count dataset)
        all-indices (vec (range num-rows))]
    (if (empty? candidates)
      nil
      (loop [survivors candidates
             sample-frac 0.25]
        (if (or (<= (count survivors) 1) (>= sample-frac 1.0))
          ;; Final evaluation on full dataset
          (let [evaluated (map #(hp/evaluate-hyperplane dataset all-indices % target-key loss-fn) survivors)
                best (reduce (fn [best current]
                               (if (or (nil? best) (> (:loss-reduction current) (:loss-reduction best)))
                                 current
                                 best))
                             nil
                             evaluated)]
            best)
          ;; Intermediate round
          (let [sample-size (max 1 (int (* num-rows sample-frac)))
                sample-indices (take sample-size (utils/seeded-shuffle rng all-indices))
                evaluated (map #(hp/evaluate-hyperplane dataset sample-indices % target-key loss-fn) survivors)
                sorted (sort-by :loss-reduction > evaluated)
                keep-count (max 1 (quot (count sorted) 2))]
            (recur (take keep-count sorted) (* sample-frac 2.0))))))))

(defn best-oblique-split
  "Finds the best oblique split using successive halving."
  [dataset continuous-features target-key loss-fn M]
  (let [rng (java.util.Random. (hash dataset))
        candidates (hp/generate-candidate-hyperplanes continuous-features dataset M rng)
        best-candidate (successive-halving-search dataset candidates target-key loss-fn rng)]
    (when (and best-candidate (> (:loss-reduction best-candidate) 0.0))
      (let [[left-ds right-ds] (ds/split-by-oblique dataset (:weights best-candidate) (:threshold best-candidate))]
        (assoc best-candidate
               :type :oblique
               :left left-ds
               :right right-ds)))))

(defn best-split
  "Finds the best split, evaluating both categorical and oblique continuous splits."
  ([dataset target-key]
   (best-split dataset target-key {}))

  ([dataset target-key {:keys [features loss-fn num-candidates]
                        :or {features nil
                             loss-fn loss/mean-squared-deviation
                             num-candidates 50}}]
   (let [available-features (or features (remove #{target-key} (ds/column-names dataset)))
         cat-features (filter #(= (ds/get-type dataset %) :categorical) available-features)
         cont-features (filter #(= (ds/get-type dataset %) :continuous) available-features)

         best-cat (->> cat-features
                       (map #(hp/best-categorical-split dataset % target-key loss-fn))
                       (remove nil?)
                       (reduce (fn [best current]
                                 (if (or (nil? best) (> (:loss-reduction current) (:loss-reduction best)))
                                   current
                                   best))
                               nil))

         best-obl (when (seq cont-features)
                    (best-oblique-split dataset cont-features target-key loss-fn num-candidates))]

     (cond
       (and best-cat best-obl) (if (> (:loss-reduction best-obl) (:loss-reduction best-cat)) best-obl best-cat)
       best-cat best-cat
       best-obl best-obl
       :else nil))))

(defn train
  "Trains an Oblique Regression tree on the given dataset."
  ([dataset target-key]
   (train dataset target-key {}))

  ([dataset target-key {:keys [task-type stop loss-fn num-candidates]
                        :or   {task-type :regression
                               stop      stopping/cart-stopping-strategy
                               num-candidates 50}
                        :as   opt}]
   (if (not= task-type :regression)
     (throw (ex-info "Oblique trees currently only support regression" {:task-type task-type}))
     (let [rng         (java.util.Random. (hash dataset))
           loss-fn     (or loss-fn (builder/default-loss-fn task-type))
           best-split-fn (fn [ds tgt o]
                           (let [feats (or (:features o) (remove #{tgt} (ds/column-names ds)))
                                 m (:max-features o)
                                 sampled (if m (vec (take m (utils/seeded-shuffle rng feats))) feats)]
                             (best-split ds tgt (assoc o :features sampled :loss-fn loss-fn :num-candidates (or num-candidates 50)))))]
       (builder/build-tree dataset target-key (assoc opt :loss-fn loss-fn :stop stop :task-type task-type) best-split-fn)))))
