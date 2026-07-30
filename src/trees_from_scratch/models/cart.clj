(ns trees-from-scratch.models.cart
  (:require [trees-from-scratch.loss :as loss]
            [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.trees.binary :as binary-tree]
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

(defn vector-splits-continuous [v labels]
  (->> (map vector v labels)
       (sort-by first)
       (partition 2 1)
       (keep (fn [[[x l1] [y l2]]]
               (when (and (not= x y) (not= l1 l2))
                 (/ (+ x y) 2))))
       distinct))

(defn vector-splits-categorical [v]
  (distinct v))

(defn vector-splits [v type labels]
  (case type
    :continuous (vector-splits-continuous v labels)
    :categorical (vector-splits-categorical v)))

(defn best-vector-split [dataset feat target-key loss-fn]
  (let [v (ds/get-column dataset feat)
        feat-type (ds/get-type dataset feat)
        parent-labels (ds/get-column dataset target-key)]
    (->> (vector-splits v feat-type parent-labels)
         (map (fn [split-val]
                (let [[left-ds right-ds] (if (= feat-type :continuous)
                                           (ds/split-by-continuous dataset feat split-val)
                                           (ds/split-by-categorical dataset feat split-val))
                      left-labels  (ds/get-column left-ds target-key)
                      right-labels (ds/get-column right-ds target-key)]
                  {:type feat-type
                   :split-val split-val
                   :loss-reduction (loss-reduction parent-labels left-labels right-labels loss-fn)
                   :left left-ds
                   :right right-ds})))
         (reduce (fn [best current]
                   (if (and (> (:loss-reduction current) 0.0)
                            (or (nil? best) (> (:loss-reduction current) (:loss-reduction best))))
                     current
                     best))
                 nil))))

(defn best-split
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

(defmulti default-loss-fn identity)
(defmethod default-loss-fn :classification [_] loss/gini)
(defmethod default-loss-fn :regression [_] loss/mean-squared-deviation)

(defmulti make-leaf-node (fn [task-type _dataset _target-key] task-type))
(defmethod make-leaf-node :classification [_ dataset target-key]
  (binary-tree/make-leaf (core/majority-class (ds/get-column dataset target-key))))
(defmethod make-leaf-node :regression [_ dataset target-key]
  (binary-tree/make-leaf (core/mean-target (ds/get-column dataset target-key))))

(defmulti make-split-node (fn [type _feature _split-val _left _right] type))
(defmethod make-split-node :continuous [_ feature split-val left right]
  (binary-tree/make-continuous-split feature split-val left right))
(defmethod make-split-node :categorical [_ feature split-val left right]
  (binary-tree/make-categorical-split feature split-val left right))

(defn train
  ([dataset target-key]
   (train dataset target-key {}))

  ([dataset target-key {:keys [task-type stop loss-fn features depth]
                        :or   {task-type :classification
                               stop      stopping/cart-stopping-strategy
                               depth     0}
                        :as   opt}]
   (let [{:keys [early-exit late-exit]} stop
         loss-fn (or loss-fn (default-loss-fn task-type))
         features    (or features (remove #{target-key} (ds/column-names dataset)))
         early?      (early-exit opt dataset target-key)
         new-split   (when-not early? (best-split dataset target-key {:features features :loss-fn loss-fn}))
         late?       (and new-split (late-exit opt new-split))]
     (if (or early? late? (nil? new-split))
       (make-leaf-node task-type dataset target-key)
       (let [{:keys [feature split-val left right type]} new-split
             next-opt   (assoc opt :depth (inc depth))
             left-node  (train left target-key next-opt)
             right-node (train right target-key next-opt)]
         (make-split-node type feature split-val left-node right-node))))))
