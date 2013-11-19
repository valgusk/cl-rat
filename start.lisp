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
        :block-dim 128
        :grid-dim (ceiling (/ (* bas-count (getf rat 'count) (getf o 'count)) 128))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;    basement generation code    ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defstruct basement rats plants cats walls)



(defkernel cu-clear-candidates (void ((candidates int*)))
  (let* ((max-i (* 100 100))
         (i (+ (* block-dim-x block-idx-x) thread-idx-x)))
    (if (< i max-i) (set (aref candidates i) 1))))

;leave cells free from objects within distance
(defkernel cu-free-from (void ((candidates int*) (distance float) (otherwise-p float)
                               (objs float*) (obj-step int) (obj-count int) (obj-start int)))
  (let* ((max-i (* obj-count (* 100 100)))
         (i (+ (* block-dim-x block-idx-x) thread-idx-x))
         (obj-i (/i 10000)) ;100x100 cells
         (pos-i (- i (* obj-i 10000)))
         (x (to-float (/ pos-i 100)))
         (y (to-float (- pos-i (* 100 x)))))
    (if (< i max-i)
      (let* ((obj-x (aref objects (+ obj-start (* obj-i obj-step))))
             (obj-y (aref objects (+ (+ obj-start 1) (* obj-i obj-step))))
             (obj-dist (fmaxf (+ (fabsf (- x obj-x))
                                 (fabsf (- y obj-y))))))
        (if (> (copysignf (- distance obj-dist) otherwise-p) 0.0)
          (set (aref candidates pos-i) 0))))))

;;;;;;;;;;;;;;;;;;;;;;; WALLS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; create walls, manually defined walls go to existing and
; should have format: (x y angle length)
(defun make-walls (&optional (count 100) (len 50) (existing nil))
  (labels
    ((wall-def nil (mapcar #'random `(100 100 ,(* PI 2) ,len)))
     (stick-p (a b) (> 2 (apply #'max (mapcar #'abs (mapcar '- a b)))))
     (by-x (angle) (member (floor (/ angle PI 1/4)) '(0 3 4 7)))
     (sticks (a) (lambda (bs) (find a bs :test #'stick-p)))
     (make-grps (crosses grps &optional (i 0) (grp (nth i grps)))
       (cond ((null crosses) grps)
             ((null (car crosses)) (make-grps (cdr crosses) grps (1+ i)))
             ((zerop grp) nil)
             (t (make-grps (cdr crosses) (substitute 0 grp grps) (1+ i)))))
     (unblock-wall (fixed w walls &optional grps ok-bricks)
       (symbol-macrolet
         ((new-grps (make-grps (mapcar (sticks (car w)) walls) grps))
          (bad-p (when walls (not new-grps)))
          (ok-grps (if (or fixed ok-bricks) (mapcar #'1+ (cons 0 grps)) grps))
          (ok-walls (if ok-bricks (cons ok-bricks walls) walls)))
         (cond (fixed (values (cons w walls) ok-grps))
               ((null w) (values ok-walls ok-grps))
               (bad-p (unblock-wall nil (cdr w) ok-walls ok-grps))
               (t (unblock-wall nil (cdr w) walls new-grps (cons (car w) ok-bricks))))))
     (make-bricks (x y angle size)
       (loop for i below size
             for x1 = x then (if (by-x angle) (1+ x1) (+ x (* i (cos angle))))
             for y1 = y then (if (by-x angle) (+ y (* i (sin angle))) (1+ y1))
             when (>= 99.4 (max x1 y1) (min x1 y1) -0.5)
             collect (list (round x1) (round y1))))
     (make-all-bricks (existing left)
       (multiple-value-call #'unblock-wall (nth left existing)
         (apply #'make-bricks (or (nth left existing) (wall-def)))
         (and (plusp left) (make-all-bricks existing (1- left))))))
    (make-all-bricks existing (+ count (list-length existing)))))

;;;;;;;;;;;;;;;;;;;;;;; WALLS END ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;; PLANTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



(defun make-plants (basement plant-count wall-count wall-blck wall-step basement-wall-start)
  (with-memory-block (candidates-blk 'int (* 100 100))
      (flet ((filter (dist sign)
               (cu-free-from
                 candidates-blk dist sign
                 wall-blck wall-step wall-count basement-wall-start
                 :grid-dim (list ,(ceiling (/ (* 100 100 wall-count) 256)) 1 1)
                 :block-dim '(256 1 1))))
        (cu-clear-candidates candidates-blk)
        (filter 0.01 1.0) ;cannot grow on wall
        (filter 0.1 -1.0) ;should have a wall nearby
        (memcpy-device-to-host candidates-blk))))

;;;;;;;;;;;;;;;;;;;;;;; PLANTS END ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;; BASEMENTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



(defun make-basements (count &key (wall-count 10) (wall-len 50))
  (loop for i below count
        for bound-walls = `((0 0 ,0 100)
                            (0 0 ,(* PI 1.5) 100)
                            (99 0 ,(* PI 1.5) 100)
                            (0 99 0 100))
        for walls = (apply #'append (make-walls wall-count wall-len bound-walls))
        collect (make-basement :walls walls)))

(defun basements-walls-to-device (basements wall-blck)
  (loop for (x y) in (apply #'append (mapcar #'basement-walls basements))
        for i = 0 then (+ 2 i) do
          (setf (mem-aref wall-blck i) x (mem-aref wall-blck (1+ i)) y)
        finally (return (memcpy-host-to-device wall-blck))))



  ; (let ((rows (loop for i from 0 to 99 collect (make-string 100 :initial-element #\.)))
  ;       (walls (make-walls 10 100
  ;                `((0 0 ,0 100)
  ;                  (0 0 ,(* PI 1.5) 100)
  ;                  (99 0 ,(* PI 1.5) 100)
  ;                  (0 99 0 100)))))
  ;     (loop for (x y) in (apply #'append walls) do
  ;       (setf (elt (nth y rows) x) #\X))
  ;   (format t "~{~a~^~%~}" rows))


(defstruct basement rats plants cats walls)

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
                         collect b))
        (available-basements (loop for b in basements
                               if (< (length (basement-rats b)) 50)
                               collect b)))
    (if (< (length ok-basements) 3)
        (nth (random (length available-basements)) available-basements)
        (nth (random (length ok-basements)) ok-basements))))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;     main application code      ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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
            (rat-dissect (i a-rat) rat-reg-0)
            (rat-dissect (i b-rat) rat-reg-1)
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
            (push rat (basement-rats ok-basement))
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
                    for stat-i from 0 below stat-count do
                (setf (getf rat stat) (mem-aref stat-blk (+ (* stat-count (i rat)) stat-i)))))
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
         (start)
         ; (run-rat)
         ; (validate)
        ))))



(main)
