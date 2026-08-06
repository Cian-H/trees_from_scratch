(ns trees-from-scratch.models.random-forest
  "Implementation of the the Random Forest algorithm."
  (:require [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.trees.ensemble :as ensemble]
            [trees-from-scratch.models.cart :as cart]
            [trees-from-scratch.utils.core :as utils]
            [trees-from-scratch.metrics.evaluation :as evaluation]))

(defn get-random-features [m features]
  (vec (take m (shuffle features))))

(defn bootstrap-with-oob [dataset]
  (let [n          (ds/row-count dataset)
        in-bag-idx (repeatedly n #(rand-int n))
        train-idx  (set in-bag-idx)
        train-ds   (ds/map-columns dataset #(mapv % in-bag-idx))
        oob-idx    (filter #(not (contains? train-idx %)) (range n))
        oob-ds     (ds/map-columns dataset #(mapv % oob-idx))]
    {:train-ds  train-ds
     :train-idx train-idx
     :oob-ds    oob-ds
     :oob-idx   oob-idx}))

(defn make-pred-matrix [dataset]
  (vec (take (ds/row-count dataset) (repeat []))))

(defn update-pred-matrix [pred-matrix pred-ds pred-idx]
  (reduce (fn [matrix [idx pred]]
            (update matrix idx conj pred))
          pred-matrix
          (map vector pred-idx pred-ds)))

(defn calculate-forest-error [pmatrix target {:keys [task-type loss-fn]}]
  (let [paired-data (map vector pmatrix target)
        valid-pairs (filter (fn [[preds _actual]] (seq preds)) paired-data)
        consensus-fn (case task-type
                       :classification (fn [preds]
                                         (key (apply max-key val (frequencies preds))))
                       :regression     (fn [preds]
                                         (/ (apply + preds) (count preds))))
        final-preds   (map (fn [[preds _actual]] (consensus-fn preds)) valid-pairs)
        final-targets (map second valid-pairs)]
    (if (empty? final-preds)
      Double/POSITIVE_INFINITY
      (loss-fn final-preds final-targets))))

(defn train
  "Trains a random forest ensemble on the given dataset."
  ([dataset target-key]
   (train dataset target-key {}))

  ([dataset target-key {:keys [task-type features window-size min-delta max-trees max-features]
                        :or {window-size 10, min-delta 0.01, max-trees 100}
                        :as opt}]
   (let [features      (or features (remove #{target-key} (ds/column-names dataset)))
         p             (count features)
         raw-m         (case task-type
                         :classification (Math/sqrt p)
                         :regression (/ p 3.0))
         m             (or max-features (min p (max (int raw-m) 2)))
         target-vector (target-key dataset)]

     (loop [forest        []
            pmatrix       (make-pred-matrix dataset)
            error-history []]

       (if (or (>= (count forest) max-trees)
               (and (>= (count error-history) window-size)
                    (let [oldest (nth error-history (- (count error-history) window-size))
                          newest (peek error-history)
                          delta  (if (zero? oldest) 0 (/ (- oldest newest) oldest))]
                      (< delta min-delta))))
         (let [agg-fn (case task-type
                        :classification utils/majority-class
                        :regression     utils/mean-target)]
           (apply ensemble/make agg-fn forest))
         (let [{:keys [train-ds oob-ds oob-idx]} (bootstrap-with-oob dataset)
               tree            (cart/train
                                train-ds
                                target-key
                                (assoc opt :max-features m))
               oob-preds       (evaluation/predict-all tree oob-ds)
               new-pmatrix     (update-pred-matrix pmatrix oob-preds oob-idx)
               new-error       (calculate-forest-error new-pmatrix target-vector opt)]
           (recur (conj forest tree)
                  new-pmatrix
                  (conj error-history new-error))))))))
