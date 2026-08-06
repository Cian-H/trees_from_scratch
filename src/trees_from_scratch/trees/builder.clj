(ns trees-from-scratch.trees.builder
  "Generic tree building abstractions."
  (:require [trees-from-scratch.trees.binary :as btree]
            [trees-from-scratch.metrics.loss :as loss]
            [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.utils.core :as utils]))

(defmulti default-loss-fn identity)
(defmethod default-loss-fn :classification [_] loss/gini)
(defmethod default-loss-fn :regression [_] loss/mean-squared-deviation)

(defmulti make-leaf-node
  "Creates a leaf node appropriate for the task type."
  (fn [task-type _dataset _target-key] task-type))
(defmethod make-leaf-node :classification [_ dataset target-key]
  (btree/make-leaf (utils/majority-class (ds/get-column dataset target-key))))
(defmethod make-leaf-node :regression [_ dataset target-key]
  (btree/make-leaf (utils/mean-target (ds/get-column dataset target-key))))

(defmulti make-split-node
  "Creates a split node appropriate for the feature type."
  (fn [type _split-info _left _right] type))
(defmethod make-split-node :continuous [_ split-info left right]
  (btree/make-continuous-split (:feature split-info) (:split-val split-info) left right))
(defmethod make-split-node :categorical [_ split-info left right]
  (btree/make-categorical-split (:feature split-info) (:split-val split-info) left right))
(defmethod make-split-node :oblique [_ split-info left right]
  (btree/make-oblique-split (:weights split-info) (:threshold split-info) left right))

(defn build-tree
  "Recursively builds a tree."
  [dataset target-key opt best-split-fn]
  (let [{:keys [task-type stop depth]
         :or {task-type :regression, depth 0}} opt
        {:keys [early-exit late-exit]} stop
        early? (early-exit opt dataset target-key)
        new-split (when-not early? (best-split-fn dataset target-key opt))
        late? (and new-split (late-exit opt new-split target-key))]
    (if (or early? late? (nil? new-split))
      (make-leaf-node task-type dataset target-key)
      (let [{:keys [left right type]} new-split
            next-opt (assoc opt :depth (inc depth))
            left-node (build-tree left target-key next-opt best-split-fn)
            right-node (build-tree right target-key next-opt best-split-fn)]
        (make-split-node type new-split left-node right-node)))))
