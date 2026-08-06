(ns trees-from-scratch.models.oblique
  "Implementation of Oblique Regression Trees using Successive Halving for split optimization."
  (:require [trees-from-scratch.loss :as loss]
            [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.trees.binary :as btree]
            [trees-from-scratch.stopping :as stopping]
            [trees-from-scratch.models.core :as core]
            [trees-from-scratch.spatial.hyperplane :as hp]))

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
                sample-indices (take sample-size (core/seeded-shuffle rng all-indices))
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

(defmulti default-loss-fn identity)
(defmethod default-loss-fn :regression [_] loss/mean-squared-deviation)

(defmulti make-leaf-node (fn [task-type _dataset _target-key] task-type))
(defmethod make-leaf-node :regression [_ dataset target-key]
  (btree/make-leaf (core/mean-target (ds/get-column dataset target-key))))

(defmulti make-split-node (fn [type _split-info _left _right] type))
(defmethod make-split-node :categorical [_ split-info left right]
  (btree/make-categorical-split (:feature split-info) (:split-val split-info) left right))
(defmethod make-split-node :oblique [_ split-info left right]
  (btree/make-oblique-split (:weights split-info) (:threshold split-info) left right))

(defn train
  "Trains an Oblique Regression tree on the given dataset."
  ([dataset target-key]
   (train dataset target-key {}))

  ([dataset target-key {:keys [task-type stop loss-fn features depth num-candidates]
                        :or   {task-type :regression
                               stop      stopping/cart-stopping-strategy
                               depth     0
                               num-candidates 50}
                        :as   opt}]
   (if (not= task-type :regression)
     (throw (ex-info "Oblique trees currently only support regression" {:task-type task-type}))
     (let [rng         (java.util.Random. (hash dataset))
           {:keys [early-exit late-exit]} stop
           loss-fn     (or loss-fn (default-loss-fn task-type))
           features    (or features (remove #{target-key} (ds/column-names dataset)))
           m           (:max-features opt)
           sampled-features (if m (vec (take m (core/seeded-shuffle rng features))) features)
           early?      (early-exit opt dataset target-key)
           new-split   (when-not early? (best-split dataset target-key {:features sampled-features :loss-fn loss-fn :num-candidates num-candidates}))
           late?       (and new-split (late-exit opt new-split target-key))]
       (if (or early? late? (nil? new-split))
         (make-leaf-node task-type dataset target-key)
         (let [{:keys [left right type]} new-split
               next-opt   (assoc opt :depth (inc depth))
               left-node  (train left target-key next-opt)
               right-node (train right target-key next-opt)]
           (make-split-node type new-split left-node right-node)))))))
