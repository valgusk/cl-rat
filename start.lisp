(ql:quickload :cl-cuda)
(ql:quickload :imago)
(ql:quickload :CL-STOPWATCH)
(ql:quickload :cl-glut)
(ql:quickload :cl-glu)
(in-package :cl-cuda)
(setf *nvcc-options* (list "-arch=sm_20" "-m32"))

(in-package :cl-user)
(defpackage genetics
  (:use :cl
        :cl-cuda)
  (:export :main))


(setf *random-state* (make-random-state t))
(in-package :genetics)

(load "geneural.lisp")

(defkernel get-obj-angle (float ((rot float) (x float) (y float) (x1 float) (y1 float)))
  (let ((negative-rot (- 0.0 rot))
        (new-x1 (- (* (- x1 x) (cosf negative-rot))
                   (* (- y1 y) (sinf negative-rot))))
        (new-y1 (+ (* (- x1 x) (sinf negative-rot))
                   (* (- y1 y) (cosf negative-rot)))))
    (return (atan2f new-x1 new-y1))))


(defkernel update-sight (float ((rot float) (x float) (y float) (x1 float) (y1 float)
                                (where float*) (start-off int) (start-deg float)
                                (end-deg float) (type float) (active float)))
  (let ((stp 0.005)
        (a (get-obj-angle rot x y (- x1 stp) (- y1 stp)))
        (b (get-obj-angle rot x y (- x1 stp) (+ y1 stp)))
        (c (get-obj-angle rot x y (+ x1 stp) (- y1 stp)))
        (d (get-obj-angle rot x y (+ x1 stp) (+ y1 stp)))
        (min (fminf (fminf a b) (fminf c d)))
        (max (fmaxf (fmaxf a b) (fmaxf c d)))
        (dist (/ (sqrt (+ (* (- x1 x) (- x1 x)) (* (- y1 y) (- y1 y)))) 2.0))
        (min-deg (* 180.0 (/ min 3.141592654)))
        (max-deg (* 180.0 (/ max 3.141592654)))
        (deg-diff (fabsf (- max-deg min-deg)))
        (off start-off))
    (do ((deg start-deg (+ deg 1.0)))
        ((> deg end-deg))
      (let ((old-dist (aref where off))
            (a-diff (copysignf 1.0 (- min-deg deg)))  ;should be < 0
            (b-diff (copysignf 1.0 (- deg max-deg))) ;should be < 0
            (is-between (fminf 0.0 (* a-diff b-diff))) ; 0,1
            (is-shorter (fminf 0.0 (copysignf 1.0 (- old-dist dist)))) ; 0,1
            (should-be-replaced (* active is-between is-shorter)))
        (set (aref where off) (+ (* old-dist (- 1.0 should-be-replaced))
                                 (* dist should-be-replaced)))
        (set (aref where (+ 1 off))
             (+ (* (aref where (+ 1 off)) (- 1.0 should-be-replaced))
                (* type should-be-replaced))))
      (set off (+ 2 off)))))



(defkernel rat-see (void ((inputs float*) (input-step int) (input-off int)
                          (rats   float*) (rat-step   int) (rat-count int)
                          (objs   float*) (obj-step   int) (obj-count int) (obj-type float)
                          (max-i  int)))
  (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (if (< i max-i)
      (let ((bas-count (* rat-count obj-count))
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
      (light-objects
        rat-a-inp inp-step inp-off
        (getf rat 'blk) (getf rat 'step) (getf rat 'count)
        (getf o 'blk) (getf o 'step) (getf o 'count) (getf o 'type)
        :block-dim 128
        :grid-dim (ceiling (/ (* bas-count (getf rat 'count) (getf o 'count)) 128))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;     main application code      ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defstruct basement rats plants cats walls)

(defun neighbour-tester (d &optional (dist 2) same-is-neighbour)
  #'(lambda (f)
      (and (< (abs (- (getf f 'x) (getf d 'x))) dist)
           (< (abs (- (getf f 'y) (getf d 'y))) dist)
           (or same-is-neighbour
               (not (and (equal (getf f 'x) (getf d 'x))
                         (equal (getf f 'y) (getf d 'y))))))))

(defun connected (dots to &optional tested)
  (if dots
      (let ((con (loop for d in dots append
                    (when (some (neighbour-tester d) to)
                          (list d)))))
        (if con
            (connected (set-difference dots con) con (append tested to))
            (append to tested)))
      (append to tested)))


(defun build-walls ()
  (let ((initial-walls `(((0 0)(0 10))
                         ((0 0)(10 0))
                         ((10 0)(10 10))
                         ((0 10)(10 10))))
        (random-walls
          (loop for i from 1 to 4 collect
            `((,(+ 2 (random 96)) ,(+ 2 (random 96)))
              (,(+ 2 (random 96)) ,(+ 2 (random 96))))))
        (wall-structure nil)
        (border-structure nil)
        (group 0))
    (loop for x from 0 to 99 do
      (push `(x ,x y 0 grp nil type #\W) border-structure)
      (push `(x ,x y 99 grp nil type #\W) border-structure))
    (loop for y from 1 to 98 do
      (push `(x ,0 y ,y grp nil type #\W) border-structure)
      (push `(x ,99 y ,y grp nil type #\W) border-structure))
    (loop for wall in (append initial-walls random-walls)  do
      (destructuring-bind ((x1 y1) (x2 y2)) wall
        (let ((by-x (> (abs (- x1 x2)) (abs (- y1 y2)))))
          (loop for i from (if by-x (min x1 x2) (min y1 y2)) to (if by-x (max x1 x2) (max y1 y2))
                for j from 1 to 80
                as x = (if by-x i (+ x1 (round (* (- x2 x1) (/ (- i y1) (- y2 y1))))))
                as y = (if by-x (+ y1 (round (* (- y2 y1) (/ (- i x1) (- x2 x1))))) i) do
            (incf group)
            (let* ((new `(x ,x y ,y grp nil type #\W))
                   (neighbours
                     (remove-if-not (neighbour-tester new) wall-structure))
                   (same (find-if #'(lambda (b) (and (equal (getf b 'x) x)
                                                     (equal (getf b 'y) y)))
                                  wall-structure)))
              (when
                (and
                 (not same)
                 (or (or (> x 20) (> y 20)) (member wall initial-walls))
                 (notany #'null
                  (loop for n in neighbours
                        as same-group = (set-difference
                                          (remove n neighbours)
                                          (remove-if #'(lambda (n2) (equal (getf n 'grp)
                                                                           (getf n2 'grp)))
                                                     neighbours))
                        as group-connected = (connected same-group (list n))
                        unless (= (length same-group) (length (remove n group-connected)))
                        collect nil)))
                (push new wall-structure)
                (loop for c in (connected (remove new wall-structure)
                                          (list (or same new))) do
                  (setf (getf c 'grp) group))))))))
    (append wall-structure border-structure)))

(defun call-cats (basement)
  (loop repeat 4 do
    (loop do
      (let* ((x (+ 20 (random 80)))
             (y (+ 20 (random 80)))
             (cat `(x ,x y ,y type #\C health 1.0)))
        (when (notany (neighbour-tester cat 4 T)
                      (append (basement-walls basement)
                              (basement-cats basement)))
          (push cat (basement-cats basement))
          (return))))))


(defun grow-plants (basement)
  (loop repeat 200 do
    (loop do
      (let* ((x (random 100))
             (y (random 100))
             (plant `(x ,x y ,y type #\@ health 1.0)))
        (when (or (some (neighbour-tester plant)
                        (append (basement-walls basement)
                                (basement-plants basement)))
                  (< (random 1.0) 0.01))
          (push plant (basement-plants basement))
          (return))))))


(defun create-random-basement ()
  (let ((basement (make-basement)))
    (setf (basement-walls basement) (build-walls))
    (grow-plants basement)
    (call-cats basement)
    basement))

(defun show-objects (basement)
  (let ((rows (loop for i from 0 to 100 collect (make-string 100 :initial-element #\Space))))
    (loop for w in (append (basement-plants basement)
                           (basement-walls basement)
                           (basement-cats basement)) do
      (setf (elt (nth (getf w 'x) rows) (getf w 'y)) (getf w 'type)))
  (format t "~{~a~^~%~}" rows)))

; (show-objects (create-random-basement))

(defun find-available-basement (basements)
  (let ((ok-basements (loop for b in basements
                         if (< (length (basement-rats b)) 40)
                         collect (list b)))
        (available-basements (loop for b in basements
                               if (< (length (basement-rats b)) 50)
                               collect (list b))))
    (if (< (length ok-basements) 3)
        (nth (random (length available-basements)) available-basements)
        (nth (random (length ok-basements)) ok-basements))))

(defun main ()
  (with-cuda-context (0)
    (with-neural-networks rat
                          2
                          ;name   inputs                outputs
                          ((A     ((nil 0 64) (G 64 96))    96)
                           (B     ((A 0 96))                96)
                           (C     ((B 0 96))                96)
                           (MEM-D ((C 0 96))                32)
                           (E     ((C 0 96) (MEM-D 0 32))   96)
                           (F     ((E 0 96))                96)
                           (G     ((F 0 96))                96))

      (labels (
        (ball-threshold (top ratio)
          (getf (first (last top (round (* ratio (rat-count))))) 'balls))

        (i (rat) (getf rat 'i))

        ;selects function to handle dead rat's
        ;genetic material while respawn based on balls collected
        (select-respawn (rat top)
          (cond
            ((>= (getf rat 'balls) (ball-threshold top 1/10)) #'revive)
            ((>= (getf rat 'balls) (ball-threshold top 3/10)) #'cross)
            ((>= (getf rat 'balls) (ball-threshold top 7/10)) #'mutate)
            (T #'randomize)))

        ;;revive the rat without modification
        (revive (top rat)
          (rat-dissect (i rat) rat-reg-0)
          (memcpy-device-to-host rat-reg-0)
          (rat-crossover rat-reg-2 rat-reg-0 rat-reg-0)
          (memcpy-host-to-device rat-reg-2)
          (rat-stitch (i rat) rat-reg-2)
          rat)

        ;;replace with a new rat, made by crossing two good rats
        (cross (top rat)
          (let* ((candidates (last top (round (* 4/10 (rat-count)))))
                 (a-rat (nth (random (length candidates)) candidates))
                 (left-candidates (remove a-rat candidates))
                 (b-rat (nth (random (length left-candidates)) left-candidates)))
            (rat-dissect (getf a-rat 'i) rat-reg-0)
            (rat-dissect (getf b-rat 'i) rat-reg-1)
            (memcpy-device-to-host rat-reg-1 rat-reg-0)
            (rat-crossover rat-reg-2 rat-reg-0 rat-reg-1)
            (memcpy-host-to-device rat-reg-2)
            (rat-stitch (i rat) rat-reg-2)
            rat))

        ;;replace with a new rat, made by mutating a good rat
        (mutate (top rat)
          (let ((candidates (last top (round (* 5/10 (rat-count))))))
            (rat-dissect (i (nth (random (length candidates)) candidates)) rat-reg-0)
            (memcpy-device-to-host rat-reg-0)
            (rat-mutation rat-reg-2 rat-reg-0)
            (memcpy-host-to-device rat-reg-2)
            (rat-stitch (i rat) rat-reg-2)
            rat))

        ;;replace with a new rat, made by random
        (randomize (top rat)
          (declare (ignore top))
          (init-fill rat-reg-0)
          (rat-crossover rat-reg-2 rat-reg-0 rat-reg-0)
          (memcpy-host-to-device rat-reg-2)
          (rat-stitch (i rat) rat-reg-2)
          rat)

        ;;put a new rat into some basement
        (position-rat (rat basements)
          (let ((ok-basement (find-available-basement basements)))
            (push (basement-rats ok-basement) rat)
            (setf (getf rat 'basement) ok-basement
                  (getf rat 'rot) 0.0
                  (getf rat 'x) 0.0
                  (getf rat 'y) 0.0)))

        ;;respawn all dead rats in top
        (respawn-rats (top basements)
          (loop for rat in top do
            (when (<= (getf rat 'health) 0)
              (funcall (select-respawn rat top) top rat)
              (setf (basement-rats (getf rat 'basement))
                      (remove rat (basement-rats (getf rat 'basement)))
                    (getf rat 'health) 1
                    (getf rat 'hurt) 0
                    (getf rat 'balls) 0)
              (position-rat rat basements)))
          top)

        ;;get top of rats
        (get-top (rats)
          (sort rats #'(lambda (a b) (< (getf a 'balls) (getf b 'balls)))))

        ;;get possible state stats of a rat
        (rat-stats () '(health hurt hunger balls x y rot))

        ;;get rat stats from device and respawn dead rats
        (update-rats (rats stat-blk basements)
          (let* ((stats (rat-stats))
                 (stat-count (length stats)))
            (memcpy-device-to-host stat-blk)
            (loop for rat in rats do
              (loop for stat in stats
                    for i from 0 below stat-count do
                (setf (getf rat stat) (mem-aref stat-blk (+ (* stat-count (i rat)) 0)))))
            (respawn-rats (get-top rats) basements)))

        ;;
        (start ()
          (let* ((rats (loop for i from 0 below (rat-count) collect
                         `(health 1.0 hurt 0.0 x 0.0 y 0.0 rot 0.0 balls 0 i ,i)))
                 (basements (loop for i from 0 below (ceiling (/ (rat-count) 40)) collect
                               (create-random-basement)))
                 (basement-wall-count (loop for b in basements sum (length (basement-walls b)))))
            (with-memory-blocks ((rat-stat-blck 'float (* (length (rat-stats)) (rat-count)))
                                 (basement-blck 'int (+ (* (length basements) ;basement count
                                                          (+  2      ;wall-off-end
                                                              50     ;each-rat-i
                                                              300    ;each-plant-i
                                                              4      ;each-cat-i
                                                              )); = 356
                                                        basement-wall-count   ;total wall count
                                                        )))

              basements)))









        )
;(start)
        ;(run-rat)
        ))))



(main)