(ns trees-from-scratch.stopping
  "Defines stopping criteria strategies for building decision trees."
  (:require [trees-from-scratch.dataset :as ds]))

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
        (let [first-lbl (first labels)]
          (every? #(= first-lbl %) labels)))))

(defn cart-late-exit?
  "Determines if tree growth should stop after evaluating splits, typically if loss reduction is insufficient."
  [options split-record]
  (let [min-loss-reduction (:min-loss-reduction options 0.0)]
    (<= (:loss-reduction split-record) min-loss-reduction)))

(def cart-stopping-strategy
  "A map containing the default CART stopping strategy predicates for early and late exits."
  {:early-exit cart-early-exit?
   :late-exit  cart-late-exit?})
