(ns trees-from-scratch.trees.stopping
  "Defines stopping criteria strategies for building decision trees."
  (:require [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.utils.core :as utils]))

(defn cart-early-exit?
  "Determines if tree growth should stop before evaluating splits based on depth, sample count, or label purity."
  [options dataset target-key]
  (let [{:keys [max-depth min-samples-split depth]
         :or {max-depth 5
              min-samples-split 2
              depth 0}} options
        labels (ds/get-column dataset target-key)]
    (or (>= depth max-depth)
        (< (ds/row-count dataset) min-samples-split)
        (= 1 (count (distinct labels))))))

(defn cart-late-exit?
  "Determines if tree growth should stop after evaluating splits, typically if loss reduction is insufficient or the split is redundant."
  [options split-record target-key]
  (let [min-loss-reduction (:min-loss-reduction options 0.0)
        insufficient-loss? (<= (:loss-reduction split-record) min-loss-reduction)
        redundant-split?   (if (= (:task-type options :classification) :classification)
                             (let [left-labels  (ds/get-column (:left split-record) target-key)
                                   right-labels (ds/get-column (:right split-record) target-key)]
                               (= (utils/majority-class left-labels)
                                  (utils/majority-class right-labels)))
                             false)]
    (or insufficient-loss? redundant-split?)))

(def cart-stopping-strategy
  "A map containing the default CART stopping strategy predicates for early and late exits."
  {:early-exit cart-early-exit?
   :late-exit  cart-late-exit?})
