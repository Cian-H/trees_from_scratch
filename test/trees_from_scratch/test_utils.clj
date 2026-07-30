(ns trees-from-scratch.test-utils)

(defn approx=
  ([a b] (approx= a b 1e-5))
  ([a b tol]
   (< (Math/abs (double (- a b))) tol)))
