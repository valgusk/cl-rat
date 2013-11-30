
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;        rat sensor code         ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;get object's [x1,y1] angle relative to [x,y] objectš current rotation
;;basically project [x1,y1] on [x,y]š angle system
(defkernel get-obj-angle (float ((rot float) (x float) (y float) (x1 float) (y1 float)))
  (let* ((negative-rot (- 0.0 rot))
         (new-x1 (- (* (- x1 x) (cosf negative-rot))
                    (* (- y1 y) (sinf negative-rot))))
         (new-y1 (+ (* (- x1 x) (sinf negative-rot))
                    (* (- y1 y) (cosf negative-rot)))))
    (return (atan2f new-x1 new-y1))))

;;update sensor memory of [x,y] to view [x1,y1] if seen
;;where is sensor memory array
;;start-off and end-off are boundary offsets in that memory
;;start-deg and end-deg are boundary degrees relative to [x,y] vision
;;type is [x1, y1] object's type
;;active defines if [x1,y1] can be seen
(defkernel update-sight (float ((rot float) (x float) (y float) (x1 float) (y1 float)
                                (where float*) (start-off int) (start-deg float)
                                (end-deg float) (type float) (active float)))
  (let* ((stp 0.005) ;fatness of object = 0.005 * 2 = 1/100 of field
         ;;get angles object covers
         (a (get-obj-angle rot x y (- x1 stp) (- y1 stp)))
         (b (get-obj-angle rot x y (- x1 stp) (+ y1 stp)))
         (c (get-obj-angle rot x y (+ x1 stp) (- y1 stp)))
         (d (get-obj-angle rot x y (+ x1 stp) (+ y1 stp)))
         (min (fminf (fminf a b) (fminf c d))) ;object's start angle
         (max (fmaxf (fmaxf a b) (fmaxf c d))) ;object's end angle
         ;; distance to object
         (dist (/ (sqrt (+ (* (- x1 x) (- x1 x)) (* (- y1 y) (- y1 y)))) 2.0))
         ;; minimum in degrees
         (min-deg (* 180.0 (/ min 3.141592654)))
         ;; maximum in degrees
         (max-deg (* 180.0 (/ max 3.141592654)))
         ;; indegree length covered by object
         (deg-diff (fabsf (- max-deg min-deg)))
         ;;offset in memory that corresponds to angle viewed
         (off start-off))
    (do ((deg start-deg (+ deg 1.0))) ;;each 1° the receptor able to see
        ((> deg end-deg))
      (let* ((old-dist (aref where off)) ;;distance present in this angle's memory cell
             (a-diff (copysignf 1.0 (- min-deg deg)))  ;should be -1
             (b-diff (copysignf 1.0 (- deg max-deg))) ;should be -1
             (is-between (fminf 0.0 (* a-diff b-diff))) ; object covers that angle? 0,1
             (is-shorter (fminf 0.0 (copysignf 1.0 (- old-dist dist)))) ; distance is shorter? 0,1
             (should-be-replaced (* active is-between is-shorter))) ;should replace vision? 0,1
        ;; replace distance if should be replaced
        (set (aref where off) (+ (* old-dist (- 1.0 should-be-replaced))
                                 (* dist should-be-replaced)))
        ;; replace type if should by replaced
        (set (aref where (+ 1 off)) ;
             (+ (* (aref where (+ 1 off)) (- 1.0 should-be-replaced))
                (* type should-be-replaced))))
      (set off (+ 2 off)))))


;;update vision for all rats for all objects of type obj-type
;;step values are sizes of objects in memory arrays
(defkernel rat-see (void ((inputs float*) (input-step int) (input-off int)
                          (rats   float*) (rat-step   int) (rat-count int)
                          (objs   float*) (obj-step   int) (obj-count int) (obj-type float)
                          (max-i  int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (if (< i max-i)
      (let* ((bas-count (* rat-count obj-count))
             (bas-i (/ i bas-count))
             (rat-i (/ (- i (* bas-count bas-i)) obj-count))
             (obj-i (- i (* bas-count bas-i) (* rat-i rat-count)))
             (rat-start (+ (* bas-i rat-step rat-count) (* rat-i rat-step)))
             (input-start (* rat-i input-step))
             (obj-start (+ (* bas-i obj-step obj-count) (* obj-i obj-step)))
             (rat-x (aref rats rat-start))
             (rat-y (aref rats (+ 1 rat-start)))
             (rat-rotation (aref rats (+ 3 rat-start)))
             (obj-x (aref objs obj-start))
             (obj-y (aref objs (+ 1 obj-start)))
             (obj-health (aref objs (+ 2 obj-start)))
             (obj-alive (fminf 0.0 (copysignf 1.0 obj-health))))
        (update-sight rat-rotation rat-x rat-y obj-x obj-y inputs input-start
                      -29.0 28.0 ; +- degrees of vision
                      obj-type
                      obj-alive)))))

(defun update-vision (rat-a-inp inp-step inp-off bas-count &rest visioned)
  (let ((rat (find-if #'(lambda (o) (= (getf o 'type) 0.5)) visioned )))
    (loop for o in visioned do
      (rat-see
        rat-a-inp inp-step inp-off
        (getf rat 'blk) (getf rat 'step) (getf rat 'count)
        (getf o 'blk) (getf o 'step) (getf o 'count) (getf o 'type)
        (* bas-count (getf o 'count) (getf rat 'count))
        :block-dim 128
        :grid-dim (ceiling (/ (* bas-count (getf rat 'count) (getf o 'count)) 128))))))
