(ns trees-from-scratch.models.oblique
  "Implementation of Oblique Regression Trees using Successive Halving for split optimization."
  (:require [trees-from-scratch.loss :as loss]
            [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.trees.binary :as btree]
            [trees-from-scratch.stopping :as stopping]
            [trees-from-scratch.models.core :as core]))

(defn loss-reduction
  [parent-labels left-labels right-labels loss-fn]
  (let [n       (count parent-labels)
        n-left  (count left-labels)
        n-right (count right-labels)]
    (if (zero? n)
      0.0
      (let [weight-left  (/ (double n-left) n)
            weight-right (/ (double n-right) n)]
        (- (loss-fn parent-labels)
           (+ (* weight-left  (loss-fn left-labels))
              (* weight-right (loss-fn right-labels))))))))

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
                              :loss-reduction (loss-reduction parent-labels left-labels right-labels loss-fn)})))
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

;; -- Oblique Splitting Logic --

(defn seeded-shuffle [^java.util.Random rng coll]
  (let [al (java.util.ArrayList. ^java.util.Collection coll)]
    (java.util.Collections/shuffle al rng)
    (vec al)))

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
    (assoc candidate :loss-reduction (loss-reduction subset-labels left-labels right-labels loss-fn))))

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
          (let [evaluated (map #(evaluate-hyperplane dataset all-indices % target-key loss-fn) survivors)
                best (reduce (fn [best current]
                               (if (or (nil? best) (> (:loss-reduction current) (:loss-reduction best)))
                                 current
                                 best))
                             nil
                             evaluated)]
            best)
          ;; Intermediate round
          (let [sample-size (max 1 (int (* num-rows sample-frac)))
                sample-indices (take sample-size (seeded-shuffle rng all-indices))
                evaluated (map #(evaluate-hyperplane dataset sample-indices % target-key loss-fn) survivors)
                sorted (sort-by :loss-reduction > evaluated)
                keep-count (max 1 (quot (count sorted) 2))]
            (recur (take keep-count sorted) (* sample-frac 2.0))))))))

(defn best-oblique-split
  "Finds the best oblique split using successive halving."
  [dataset continuous-features target-key loss-fn M]
  (let [rng (java.util.Random. (hash dataset))
        candidates (generate-candidate-hyperplanes continuous-features dataset M rng)
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
                       (map #(best-categorical-split dataset % target-key loss-fn))
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
           sampled-features (if m (vec (take m (seeded-shuffle rng features))) features)
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
