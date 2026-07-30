(ns trees-from-scratch.stopping
  (:require [trees-from-scratch.dataset :as ds]))

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
  (let [min-loss-reduction (:min-loss-reduction options 0.0)]
    (<= (:loss-reduction split-record) min-loss-reduction)))

(def cart-stopping-strategy
  {:early-exit cart-early-exit?
   :late-exit  cart-late-exit?})
