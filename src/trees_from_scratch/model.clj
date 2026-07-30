(ns trees-from-scratch.model
  (:require [trees-from-scratch.loss :as loss]
            [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.tree :as tree]))

(defn majority-class
  [labels]
  (when (seq labels)
    (->> labels
         frequencies
         (sort-by val >)
         ffirst)))

(defn mean-target
  [values]
  (when values
    (if (empty? values)
      0.0
      (/ (apply + values) (count values)))))

(defn impurity-gain
  [parent-labels left-labels right-labels impurity-fn]
  (let [n       (count parent-labels)
        n-left  (count left-labels)
        n-right (count right-labels)]
    (if (zero? n)
      0.0
      (let [weight-left  (/ (double n-left) n)
            weight-right (/ (double n-right) n)]
        (- (impurity-fn parent-labels)
           (+ (* weight-left  (impurity-fn left-labels))
              (* weight-right (impurity-fn right-labels))))))))

(defn vector-splits-continuous [v labels]
  (->> (map vector v labels)
       (sort-by first)
       (partition 2 1)
       (keep (fn [[[x l1] [y l2]]]
               ; Optimisation: it can be mathematically proven that the optimal split must fall
               ;   between 2 points with different labels
               (when (and (not= x y) (not= l1 l2))
                 (/ (+ x y) 2))))
       distinct))

(defn vector-splits-categorical [v]
  (distinct v))

(defn vector-splits [v type labels]
  (case type
    :continuous (vector-splits-continuous v labels)
    :categorical (vector-splits-categorical v)))

(defn best-vector-split [dataset feat target-key impurity-fn]
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
                   :gain (impurity-gain parent-labels left-labels right-labels impurity-fn)
                   :left left-ds
                   :right right-ds})))
         (reduce (fn [best current]
                   (if (and (> (:gain current) 0.0)
                            (or (nil? best) (> (:gain current) (:gain best))))
                     current
                     best))
                 nil))))

(defn best-split

  ([dataset target-key]
   (best-split dataset target-key {}))

  ([dataset target-key {:keys [features impurity-fn]
                        :or {features nil
                             impurity-fn loss/gini}}]
   (let [available-features (or features (remove #{target-key} (ds/column-names dataset)))]
     (->> available-features
          (map (fn [feat]
                 (let [best-split-for-feat (best-vector-split dataset feat target-key impurity-fn)]
                   (when best-split-for-feat
                     (assoc best-split-for-feat :feature feat)))))
          (remove nil?)
          (reduce (fn [best current]
                    (if (or (nil? best) (> (:gain current) (:gain best)))
                      current
                      best))
                  nil)))))

(defn cart-early-exit?
  [options dataset target-key]
  (let [{:keys [max-depth min-samples-split depth]
         :or {max-depth 5
              min-samples-split 2
              depth 0}} options
        labels (ds/get-column dataset target-key)]
    (or (>= depth max-depth)
        (< (ds/row-count dataset) min-samples-split)
        (<= (count (set labels)) 1))))

(defn cart-late-exit?
  [options split-record]
  (let [min-gain (:min-gain options 0.0)]
    (<= (:gain split-record) min-gain)))

(def cart-stopping-strategy
  {:early-exit cart-early-exit?
   :late-exit  cart-late-exit?})

(defmulti default-impurity-fn identity)
(defmethod default-impurity-fn :classification [_] loss/gini)
(defmethod default-impurity-fn :regression [_] loss/mean-squared-deviation)

(defmulti make-leaf-node (fn [task-type _dataset _target-key] task-type))
(defmethod make-leaf-node :classification [_ dataset target-key]
  (tree/make-leaf (majority-class (ds/get-column dataset target-key))))
(defmethod make-leaf-node :regression [_ dataset target-key]
  (tree/make-leaf (mean-target (ds/get-column dataset target-key))))

(defmulti make-split-node (fn [type _feature _split-val _left _right] type))
(defmethod make-split-node :continuous [_ feature split-val left right]
  (tree/make-continuous-split feature split-val left right))
(defmethod make-split-node :categorical [_ feature split-val left right]
  (tree/make-categorical-split feature split-val left right))

(defn train

  ([dataset target-key]
   (train dataset target-key {}))

  ([dataset target-key {:keys [task-type stop impurity-fn features depth]
                        :or   {task-type :classification
                               stop      cart-stopping-strategy
                               depth     0}
                        :as   opt}]
   (let [{:keys [early-exit late-exit]} stop
         impurity-fn (or impurity-fn (default-impurity-fn task-type))
         features    (or features (remove #{target-key} (ds/column-names dataset)))
         early?      (early-exit opt dataset target-key)
         new-split   (when-not early? (best-split dataset target-key {:features features :impurity-fn impurity-fn}))
         late?       (and new-split (late-exit opt new-split))]
     (if (or early? late? (nil? new-split))
       (make-leaf-node task-type dataset target-key)
       (let [{:keys [feature split-val left right type]} new-split
             next-opt   (assoc opt :depth (inc depth))
             left-node  (train left target-key next-opt)
             right-node (train right target-key next-opt)]
         (make-split-node type feature split-val left-node right-node))))))

(defn predict-all
  [tree dataset]
  (map (fn [idx]
         (tree/predict tree (ds/get-row dataset idx)))
       (range (ds/row-count dataset))))

(defn evaluate
  ([tree dataset target-key]
   (evaluate tree dataset target-key :accuracy))
  ([tree dataset target-key metric]
   (let [predictions (predict-all tree dataset)
         actuals     (ds/get-column dataset target-key)]
     (if (empty? actuals)
       0.0
       (case metric
         :accuracy (let [correct (filter true? (map = predictions actuals))]
                     (/ (double (count correct)) (count actuals)))
         :mse      (let [errors (map (fn [p a] (let [d (- p a)] (* d d))) predictions actuals)]
                     (/ (apply + errors) (count actuals))))))))
