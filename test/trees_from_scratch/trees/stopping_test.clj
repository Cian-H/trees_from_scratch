(ns trees-from-scratch.trees.stopping-test
  (:require [clojure.test :refer [deftest is testing]]
            [trees-from-scratch.trees.stopping :as stopping]))

(def classification-dataset
  {:columns {:x [1 2 3 4]
             :label ["Yes" "No" "Yes" "No"]}
   :types {:x :continuous :label :categorical}})

(deftest test-cart-early-exit?
  (testing "stopping due to reaching max depth"
    (is (true? (stopping/cart-early-exit? {:max-depth 5 :depth 5} classification-dataset :label)))
    (is (false? (stopping/cart-early-exit? {:max-depth 5 :depth 2} classification-dataset :label))))

  (testing "stopping due to minimum samples split threshold"
    (let [small-dataset {:columns {:x [1] :label ["A"]} :types {:x :continuous :label :categorical}}]
      (is (true? (stopping/cart-early-exit? {:min-samples-split 2 :depth 1} small-dataset :label))))
    (is (false? (stopping/cart-early-exit? {:min-samples-split 2 :depth 1} classification-dataset :label))))

  (testing "stopping when dataset target column is pure"
    (let [pure-data {:columns {:x [1 2] :label ["Yes" "Yes"]} :types {:x :continuous :label :categorical}}]
      (is (true? (stopping/cart-early-exit? {:depth 0} pure-data :label))))))

(deftest test-cart-late-exit?
  (let [dummy-split {:left {:columns {:label ["A"]}}
                     :right {:columns {:label ["B"]}}}]
    (testing "returns true if loss reduction is less than or equal to min-loss-reduction"
      (let [split-record (assoc dummy-split :feature :x :loss-reduction 0.05)
            options      {:min-loss-reduction 0.1}]
        (is (true? (stopping/cart-late-exit? options split-record :label))))
      (let [split-record (assoc dummy-split :feature :x :loss-reduction 0.1)
            options      {:min-loss-reduction 0.1}]
        (is (true? (stopping/cart-late-exit? options split-record :label)))))

    (testing "returns false if loss reduction is greater than min-loss-reduction"
      (let [split-record (assoc dummy-split :feature :x :loss-reduction 0.15)
            options      {:min-loss-reduction 0.1}]
        (is (false? (stopping/cart-late-exit? options split-record :label)))))))
