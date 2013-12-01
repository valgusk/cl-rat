(defun clear-candidates (candidates-blk)
  (init-fill candidates-blk #'(lambda () 1)))


;filter cells free from objs within distance
(defkernel cu-free-from (void ((candidates int*) (distance float) (otherwise-p float)
                               (objs float*) (obj-step int) (obj-count int) (obj-start int)))
  (let* ((max-i (* obj-count (* 100 100)))
         (i (+ (* block-dim-x block-idx-x) thread-idx-x))
         (obj-i (/ i 10000)) ;100x100 cells
         (pos-i (- i (* obj-i 10000)))
         (x (/ pos-i 100)) ;incorrect value??????????????????????????
         (y (- pos-i (* 100 x))))
    (if (< i max-i)
      (let* ((obj-x (aref objs (+ obj-start (* obj-i obj-step))))
             (obj-y (aref objs (+ (+ obj-start 1) (* obj-i obj-step))))
             (obj-hp (aref objs (+ (+ obj-start 2) (* obj-i obj-step))))
             (obj-dist (fmaxf (fabsf (- (to-float x) obj-x))
                              (fabsf (- (to-float y) obj-y)))))
        (if (> obj-hp 0.0)
          (if (> (* (- distance obj-dist) otherwise-p) 0.0)
            (set (aref candidates pos-i) 0)))))))

;;destructively picks n elements from lst
(defun nrandom-pick (n lst &optional done &key (default (lambda ())))
  (if (plusp n)
    (let ((random-value (if (cdr lst)
                            (pop (nthcdr (max 1 (random (list-length lst))) lst))
                            (pop lst))))
      (nrandom-pick (1- n) lst (cons (or random-value (funcall default)) done)))
      done))

;;
(defmacro with-position-filters (filter-count filter-datas &rest body)
  `(with-memory-block (candidates-blk 'int (* 100 100))
     (labels ,(loop for (filter-name count blck step start) in filter-datas collect
                `(,filter-name (dist sign)
                    (cu-free-from
                      candidates-blk dist sign
                      ,blck ,step ,count ,start
                      :grid-dim (list (ceiling (/ (* 100 100 ,count) 256)) 1 1)
                      :block-dim '(256 1 1))))
        (clear-candidates candidates-blk)
        ,@body
        (memcpy-device-to-host candidates-blk)
        (loop for i below 10000
              for x-y = (list (floor (/ i 100)) (rem i 100))
              when (plusp (mem-aref candidates-blk i))
              collect x-y into ret
              finally (return (nrandom-pick ,filter-count ret))))))

;;visualize object placements
(defun show-basement (basement)
  (let ((rows (loop for i below 100 collect (make-string 100 :initial-element #\Space))))
    (loop for type-positions in (list #'basement-walls #'basement-cats #'basement-rats #'basement-plants)
          for type in '(#\# #\. #\r #\Y) do
      (loop for (x y) in (funcall type-positions basement)
        when  (and x y)  do (setf (elt (nth x rows) y) type)))
    (format t "~%~{~a~^~%~}~%" rows)))