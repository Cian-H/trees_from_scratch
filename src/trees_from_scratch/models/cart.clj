(ns trees-from-scratch.models.cart
  "Implementation of the CART (Classification and Regression Trees) algorithm."
  (:require [trees-from-scratch.loss :as loss]
            [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.trees.binary :as btree]
            [trees-from-scratch.stopping :as stopping]
            [trees-from-scratch.models.core :as core]))

(defn loss-reduction
  "Calculates the reduction in loss (e.g., information gain) achieved by a split."
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

(defn vector-splits-continuous
  "Finds all possible continuous split points (midpoints between distinct values) for a feature vector."
  [v labels]
  (->> (map vector v labels)
       (sort-by first)
       (partition 2 1)
       (keep (fn [[[x l1] [y l2]]]
               (when (and (not= x y) (not= l1 l2))
                 (/ (+ x y) 2))))
       distinct))

(defn vector-splits-categorical
  "Finds all possible categorical split points (unique categories) for a feature vector."
  [v]
  (distinct v))

(defn vector-splits
  "Dispatches to the appropriate split-finding function based on feature type."
  [v type labels]
  (case type
    :continuous (vector-splits-continuous v labels)
    :categorical (vector-splits-categorical v)))

(defmulti partition-labels
  "Partitions labels into left and right subsets based on a feature split."
  (fn [type _v _labels _split-val] type))

(defmethod partition-labels :continuous [_ v labels split-val]
  (reduce (fn [[l r] [val label]]
            (if (<= val split-val)
              [(conj l label) r]
              [l (conj r label)]))
          [[] []]
          (map vector v labels)))

(defmethod partition-labels :categorical [_ v labels split-val]
  (reduce (fn [[l r] [val label]]
            (if (= val split-val)
              [(conj l label) r]
              [l (conj r label)]))
          [[] []]
          (map vector v labels)))

(defn best-vector-split
  "Finds the best split (maximizing loss reduction) for a single feature."
  [dataset feat target-key loss-fn]
  (let [v (ds/get-column dataset feat)
        feat-type (ds/get-type dataset feat)
        parent-labels (ds/get-column dataset target-key)]
    (when-let [best-split-val
               (->> (vector-splits v feat-type parent-labels)
                    (map (fn [split-val]
                           (let [[left-labels right-labels] (partition-labels feat-type v parent-labels split-val)]
                             {:split-val split-val
                              :loss-reduction (loss-reduction parent-labels left-labels right-labels loss-fn)})))
                    (reduce (fn [best current]
                              (if (and (> (:loss-reduction current) 0.0)
                                       (or (nil? best) (> (:loss-reduction current) (:loss-reduction best))))
                                current
                                best))
                            nil))]
      (let [[left-ds right-ds] (if (= feat-type :continuous)
                                 (ds/split-by-continuous dataset feat (:split-val best-split-val))
                                 (ds/split-by-categorical dataset feat (:split-val best-split-val)))]
        (assoc best-split-val
               :type feat-type
               :left left-ds
               :right right-ds)))))

(defn best-split
  "Finds the best split across all available features."
  ([dataset target-key]
   (best-split dataset target-key {}))

  ([dataset target-key {:keys [features loss-fn]
                        :or {features nil
                             loss-fn loss/gini}}]
   (let [available-features (or features (remove #{target-key} (ds/column-names dataset)))]
     (->> available-features
          (map (fn [feat]
                 (let [best-split-for-feat (best-vector-split dataset feat target-key loss-fn)]
                   (when best-split-for-feat
                     (assoc best-split-for-feat :feature feat)))))
          (remove nil?)
          (reduce (fn [best current]
                    (if (or (nil? best) (> (:loss-reduction current) (:loss-reduction best)))
                      current
                      best))
                  nil)))))

(defmulti default-loss-fn
  "Returns the default loss function for a given task type."
  identity)
(defmethod default-loss-fn :classification [_] loss/gini)
(defmethod default-loss-fn :regression [_] loss/mean-squared-deviation)

(defmulti make-leaf-node
  "Creates a leaf node appropriate for the task type."
  (fn [task-type _dataset _target-key] task-type))
(defmethod make-leaf-node :classification [_ dataset target-key]
  (btree/make-leaf (core/majority-class (ds/get-column dataset target-key))))
(defmethod make-leaf-node :regression [_ dataset target-key]
  (btree/make-leaf (core/mean-target (ds/get-column dataset target-key))))

(defmulti make-split-node
  "Creates a split node appropriate for the feature type."
  (fn [type _feature _split-val _left _right] type))
(defmethod make-split-node :continuous [_ feature split-val left right]
  (btree/make-continuous-split feature split-val left right))
(defmethod make-split-node :categorical [_ feature split-val left right]
  (btree/make-categorical-split feature split-val left right))

(defn train
  "Trains a CART decision tree on the given dataset."
  ([dataset target-key]
   (train dataset target-key {}))

  ([dataset target-key {:keys [task-type stop loss-fn features depth]
                        :or   {task-type :classification
                               stop      stopping/cart-stopping-strategy
                               depth     0}
                        :as   opt}]
   (let [{:keys [early-exit late-exit]} stop
         loss-fn     (or loss-fn (default-loss-fn task-type))
         features    (or features (remove #{target-key} (ds/column-names dataset)))
         early?      (early-exit opt dataset target-key)
         new-split   (when-not early? (best-split dataset target-key {:features features :loss-fn loss-fn}))
         late?       (and new-split (late-exit opt new-split target-key))]
     (if (or early? late? (nil? new-split))
       (make-leaf-node task-type dataset target-key)
       (let [{:keys [feature split-val left right type]} new-split
             next-opt   (assoc opt :depth (inc depth))
             left-node  (train left target-key next-opt)
             right-node (train right target-key next-opt)]
         (make-split-node type feature split-val left-node right-node))))))
