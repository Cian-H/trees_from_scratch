(ns trees-from-scratch.models.gradient-boosted
  "Implementation of the the Gradient Boosted Decision Tree algorithm."
  (:require [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.utils.maths :refer [matrix-sub]]
            [trees-from-scratch.utils.core :as utils]
            [trees-from-scratch.trees.ensemble :as ensemble]
            [trees-from-scratch.models.cart :as cart]
            [trees-from-scratch.metrics.evaluation :as evaluation]))

(defn calculate-residuals [preds target]
  (matrix-sub target preds))

(defn gb-agg [learning-rate]
  (fn [preds]
    (apply + (map #(* learning-rate %) preds))))

(defn train
  "Trains a gradient boosted decision tree on the given dataset."
  ([dataset target-key]
   (train dataset target-key {}))

  ([dataset target-key {:keys [task-type learning-rate max-trees]
                        :or {learning-rate 0.1, max-trees 10}
                        :as opt}]
   (if (not (= task-type :regression))
     (throw (ex-info "gradient-boosted trees can only be used for regression"
                     {:error/type   :invalid-argument
                      :error/reason :incompatible-feature-type
                      :task/type    task-type}))
     (let [opt (utils/rename-key (merge {:subtree-depth 5} opt) :subtree-depth :tree-depth)]
       (loop [current-ds (ds/rename-columns dataset {target-key :residuals})
              trees      []]
         (if (< (count trees) max-trees)
           (let [tree         (cart/train current-ds :residuals opt)
                 preds        (evaluation/predict-all tree current-ds)
                 scaled-preds (map #(* learning-rate %) preds)
                 res-in       (ds/get-column current-ds :residuals)
                 res-out      (calculate-residuals scaled-preds res-in)]
             (recur
              (ds/replace-column current-ds :residuals res-out :continuous)
              (conj trees tree)))
           (apply ensemble/make-parallel (gb-agg learning-rate) trees)))))))
