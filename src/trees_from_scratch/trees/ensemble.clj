(ns trees-from-scratch.trees.ensemble
  "Unified entry point for tree ensemble models (parallel and cascade)."
  (:require [trees-from-scratch.trees.ensemble.parallel :as parallel]
            [trees-from-scratch.trees.ensemble.cascade :as cascade])
  (:import [trees_from_scratch.trees.ensemble.parallel ParallelEnsemble]))

(def Ensemble ParallelEnsemble)

(defn make-parallel
  "Creates a parallel tree ensemble."
  ([agg & trees]
   (apply parallel/make agg trees))

  ([agg trees]
   (apply parallel/make agg trees)))

(defn make-cascade
  "Creates a cascade (sequential) tree ensemble."
  ([agg & trees]
   (apply cascade/make agg trees))

  ([agg trees]
   (apply cascade/make agg trees))

  ([agg cascade-fn trees]
   (apply cascade/make agg cascade-fn trees))

  ([agg cascade-fn & trees]
   (apply cascade/make agg cascade-fn trees)))

(defn make
  "Creates an ensemble model. Defaults to parallel ensemble for backward compatibility."
  ([agg & trees]
   (apply make-parallel agg trees)))
