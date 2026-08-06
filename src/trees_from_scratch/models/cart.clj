(ns trees-from-scratch.models.cart
  "Implementation of the CART (Classification and Regression Trees) algorithm."
  (:require [trees-from-scratch.metrics.loss :as loss]
            [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.trees.stopping :as stopping]
            [trees-from-scratch.metrics.evaluation :as evaluation]
            [trees-from-scratch.trees.builder :as builder]))

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
                              :loss-reduction (evaluation/loss-reduction parent-labels left-labels right-labels loss-fn)})))
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
               :feature feat
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
                 (best-vector-split dataset feat target-key loss-fn)))
          (remove nil?)
          (reduce (fn [best current]
                    (if (or (nil? best) (> (:loss-reduction current) (:loss-reduction best)))
                      current
                      best))
                  nil)))))

(defn train
  "Trains a CART decision tree on the given dataset."
  ([dataset target-key]
   (train dataset target-key {}))

  ([dataset target-key {:keys [task-type stop loss-fn]
                        :or   {task-type :classification
                               stop      stopping/cart-stopping-strategy}
                        :as   opt}]
   (let [loss-fn (or loss-fn (builder/default-loss-fn task-type))
         best-split-fn (fn [ds tgt o]
                         (let [feats (or (:features o) (remove #{tgt} (ds/column-names ds)))
                               m (:max-features o)
                               sampled (if m (vec (take m (shuffle feats))) feats)]
                           (best-split ds tgt (assoc o :features sampled :loss-fn loss-fn))))]
     (builder/build-tree dataset target-key (assoc opt :loss-fn loss-fn :stop stop :task-type task-type) best-split-fn))))
