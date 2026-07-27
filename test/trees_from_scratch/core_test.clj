; (ns trees-from-scratch.core-test
;   (:require [clojure.test :refer [deftest is testing]]
;             [trees-from-scratch.core :as core]))
;
; (deftest test-make-node
;   (testing "creating a tree node"
;     (let [node (core/make-node 10)]
;       (is (= 10 (:val node)))
;       (is (nil? (:left node)))
;       (is (nil? (:right node))))))
;
; (deftest test-bst-operations
;   (testing "inserting and searching values in a BST"
;     (let [tree (reduce core/insert nil [5 3 7 1 4])]
;       (is (true? (core/contains-val? tree 5)))
;       (is (true? (core/contains-val? tree 3)))
;       (is (true? (core/contains-val? tree 7)))
;       (is (true? (core/contains-val? tree 1)))
;       (is (true? (core/contains-val? tree 4)))
;       (is (false? (core/contains-val? tree 10)))
;       (is (= [1 3 4 5 7] (core/in-order tree)))
;       (is (= 3 (core/tree-depth tree))))))
;
; (deftest test-empty-tree
;   (testing "operations on empty tree"
;     (is (false? (core/contains-val? nil 5)))
;     (is (nil? (core/in-order nil)))
;     (is (= 0 (core/tree-depth nil)))))
