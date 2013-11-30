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
(load "sensors.lisp")
(load "basement.lisp")
(print "done")

;;list of data available for rats on device
(defvar +rat-inputs+ ;rats should not know their balls value, so not included here
  `(x y health hurt hunger rot 
      ,@(loop for i from -28 to 28 
              collect (read-from-string (format nil "vision-distance-~a" i))
              collect (read-from-string (format nil "vision-object-type-~a" i)))))




(defmacro with-stats (rat-count wall-count cat-count plant-count &rest body)
  ;;lists of data that should be available on device about objects
  (let ((rat-stats '(x y health))
	    (wall-stats '(x y health))
	    (cat-stats '(x y health))
	    (plant-stats '(x y health)))
  	`(let ((rat-step ,(length (rat-stats)))
           (wall-step ,(length (wall-stats)))
           (cat-step ,(length (cat-stats)))
           (plant-step ,(length (plant-stats))))
      	(with-memory-blocks ((rat-stat-blk 'float (* rat-step ,rat-count))
                             (wall-stat-blk 'float (* wall-step ,wall-count))
                             (cat-stat-blk 'float (* cat-step ,cat-count))
                             (plant-stat-blk 'float (* plant-step ,plant-count))))
       	  ,@body)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;     main application code      ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; (defun main ()
;   (with-cuda-context (0)
;     (with-neural-networks rat
;                           2
;                           ;name   inputs                outputs
;                           ((A     ((nil 0 64) (G 64 96))    96)
;                            (B     ((A 0 96))                96)
;                            (C     ((B 0 96))                96)
;                            (MEM-D ((C 0 96))                32)
;                            (E     ((C 0 96) (MEM-D 0 32))   96)
;                            (F     ((E 0 96))                96)
;                            (G     ((F 0 96))                96))

;       (labels (
;         (ball-threshold (top ratio)
;           (getf (first (last top (round (* ratio (rat-count))))) 'balls))

;         (i (rat) (getf rat 'i))

;         ;selects function to handle dead rat's
;         ;genetic material while respawn based on balls collected
;         (select-respawn (rat top)
;           (cond
;             ((>= (getf rat 'balls) (ball-threshold top 1/10)) #'revive)
;             ((>= (getf rat 'balls) (ball-threshold top 3/10)) #'cross)
;             ((>= (getf rat 'balls) (ball-threshold top 7/10)) #'mutate)
;             (T #'randomize)))

;         ;;revive the rat without modification
;         (revive (top rat)
;           (rat-dissect (i rat) rat-reg-0)
;           (memcpy-device-to-host rat-reg-0)
;           (rat-crossover rat-reg-2 rat-reg-0 rat-reg-0)
;           (memcpy-host-to-device rat-reg-2)
;           (rat-stitch (i rat) rat-reg-2)
;           rat)

;         ;;replace with a new rat, made by crossing two good rats
;         (cross (top rat)
;           (let* ((candidates (last top (round (* 4/10 (rat-count)))))
;                  (a-rat (nth (random (length candidates)) candidates))
;                  (left-candidates (remove a-rat candidates))
;                  (b-rat (nth (random (length left-candidates)) left-candidates)))
;             (rat-dissect (i a-rat) rat-reg-0)
;             (rat-dissect (i b-rat) rat-reg-1)
;             (memcpy-device-to-host rat-reg-1 rat-reg-0)
;             (rat-crossover rat-reg-2 rat-reg-0 rat-reg-1)
;             (memcpy-host-to-device rat-reg-2)
;             (rat-stitch (i rat) rat-reg-2)
;             rat))

;         ;;replace with a new rat, made by mutating a good rat
;         (mutate (top rat)
;           (let ((candidates (last top (round (* 5/10 (rat-count))))))
;             (rat-dissect (i (nth (random (length candidates)) candidates)) rat-reg-0)
;             (memcpy-device-to-host rat-reg-0)
;             (rat-mutation rat-reg-2 rat-reg-0)
;             (memcpy-host-to-device rat-reg-2)
;             (rat-stitch (i rat) rat-reg-2)
;             rat))

;         ;;replace with a new rat, made by random
;         (randomize (top rat)
;           (declare (ignore top))
;           (init-fill rat-reg-0)
;           (rat-crossover rat-reg-2 rat-reg-0 rat-reg-0)
;           (memcpy-host-to-device rat-reg-2)
;           (rat-stitch (i rat) rat-reg-2)
;           rat)

;         ;;put a new rat into some basement
;         (position-rat (rat basements)
;           (let ((ok-basement (find-available-basement basements)))
;             (push rat (basement-rats ok-basement))
;             (setf (getf rat 'basement) ok-basement
;                   (getf rat 'rot) 0.0
;                   (getf rat 'x) 0.0
;                   (getf rat 'y) 0.0)))

;         ;;respawn all dead rats in top
;         (respawn-rats (top basements)
;           (loop for rat in top do
;             (when (<= (getf rat 'health) 0)
;               (funcall (select-respawn rat top) top rat)
;               (setf (basement-rats (getf rat 'basement))
;                       (remove rat (basement-rats (getf rat 'basement)))
;                     (getf rat 'health) 1
;                     (getf rat 'hurt) 0
;                     (getf rat 'balls) 0)
;               (position-rat rat basements)))
;           top)

;         ;;get top of rats
;         (get-top (rats)
;           (sort rats #'(lambda (a b) (< (getf a 'balls) (getf b 'balls)))))

;         ;;get possible state stats of a rat
;         (rat-stats () '(health hurt hunger balls x y rot))

;         ;;get rat stats from device and respawn dead rats
;         (update-rats (rats stat-blk basements)
;           (let* ((stats (rat-stats))
;                  (stat-count (length stats)))
;             (memcpy-device-to-host stat-blk)
;             (loop for rat in rats do
;               (loop for stat in stats
;                     for stat-i from 0 below stat-count do
;                 (setf (getf rat stat) (mem-aref stat-blk (+ (* stat-count (i rat)) stat-i)))))
;             (respawn-rats (get-top rats) basements)))

;         ;;
;         (start ()
;           (let* ((rats (loop for i from 0 below (rat-count) collect
;                          `(health 1.0 hurt 0.0 x 0.0 y 0.0 rot 0.0 balls 0 i ,i)))
;                  (basements (loop for i from 0 below (ceiling (/ (rat-count) 40)) collect
;                                (create-random-basement)))
;                  (basement-wall-count (loop for b in basements sum (length (basement-walls b)))))
;             (with-memory-blocks ((rat-stat-blck 'float (* (length (rat-stats)) (rat-count)))
;                                  (basement-blck 'int (+ (* (length basements) ;basement count
;                                                           (+  2      ;wall-off-end
;                                                               50     ;each-rat-i
;                                                               300    ;each-plant-i
;                                                               4      ;each-cat-i
;                                                               )); = 356
;                                                         basement-wall-count   ;total wall count
;                                                         )))

;               basements)))

;         )
;          (start)
;          ; (run-rat)
;          ; (validate)
;         ))))



; (main)
