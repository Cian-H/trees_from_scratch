(ns trees-from-scratch.utils.core-test)

(defn approx=
  ([a b] (approx= a b 1e-5))
  ([a b tol]
   (< (Math/abs (double (- a b))) tol)))

(defmacro with-temp-file
  [[path-binding [prefix suffix]] & body]
  `(let [temp-file# (java.io.File/createTempFile ~prefix ~suffix)
         ~path-binding (.getAbsolutePath temp-file#)]
     (try
       ~@body
       (finally
         (.delete temp-file#)))))
