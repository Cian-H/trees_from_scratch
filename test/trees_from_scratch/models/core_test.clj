(ns trees-from-scratch.models.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [trees-from-scratch.models.core :as core]
            [trees-from-scratch.test-utils :refer [approx=]]))

(deftest test-majority-class
  (testing "finding majority class label in a target vector"
    (is (= "Yes" (core/majority-class ["Yes" "No" "Yes" "Yes" "No"])))
    (is (= "A"   (core/majority-class ["A" "A" "A"]))))

  (testing "handling single element and empty input"
    (is (= "X" (core/majority-class ["X"])))
    (is (nil?  (core/majority-class [])))))

(deftest test-mean-target
  (testing "calculating mean target value for regression vector"
    (is (approx= 20.0 (core/mean-target [10.0 20.0 30.0])))
    (is (approx= 5.5 (core/mean-target [5.5]))))

  (testing "empty target vector defaults to 0.0"
    (is (approx= 0.0 (core/mean-target [])))))
