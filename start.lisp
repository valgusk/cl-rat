(in-package :cl-user)
(defpackage genetics
  (:use :cl
        :cl-cuda)
  (:export :main))


(setf *random-state* (make-random-state t))
(in-package :genetics)

(load "geneural.lisp")
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;     main application code      ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defstruct basement rats plants cats walls)

(defun neighbour-tester (d &optional (dist 2))
  #'(lambda (f)
      (and (< (abs (- (getf f 'x) (getf d 'x))) dist)
           (< (abs (- (getf f 'y) (getf d 'y))) dist)
           (not (and (equal (getf f 'x) (getf d 'x))
                     (equal (getf f 'y) (getf d 'y)))))))

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
            `((,(random 100) ,(random 100))
              (,(random 100) ,(random 100)))))
        (wall-structure nil)
        (group 0))
    (loop for wall in (append initial-walls random-walls)  do
      (destructuring-bind ((x1 y1) (x2 y2)) wall
        (let ((by-x (> (abs (- x1 x2)) (abs (- y1 y2)))))
          (loop for i from (if by-x (min x1 x2) (min y1 y2)) to (if by-x (max x1 x2) (max y1 y2))
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
    wall-structure))

(defun call-cats (basement)
  (loop repeat 4 do
    (loop do
      (let* ((x (+ 20 (random 80)))
             (y (+ 20 (random 80)))
             (cat `(x ,x y ,y type #\C)))
        (when (notany (neighbour-tester cat 4)
                      (append (basement-walls basement)
                              (basement-cats basement)))
          (push cat (basement-cats basement))
          (return))))))


(defun grow-plants (basement)
  (loop repeat 200 do
    (loop do
      (let* ((x (random 100))
             (y (random 100))
             (plant `(x ,x y ,y type #\@)))
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
  (format t "~{~a~^|~%~}" rows)))

; (show-objects (create-random-basement))






(defun basement-available (basement)
  (< (length (basement-rats basement)) 40))

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

        (select-respawn (rat top)
          (cond
            ((>= (getf rat 'balls) (ball-threshold top 1/10)) #'revive)
            ((>= (getf rat 'balls) (ball-threshold top 3/10)) #'cross)
            ((>= (getf rat 'balls) (ball-threshold top 7/10)) #'mutate)
            (T #'randomize)))

        (revive (top rat)
          (rat-dissect (i rat) rat-reg-0)
          (memcpy-device-to-host rat-reg-0)
          (rat-crossover rat-reg-2 rat-reg-0 rat-reg-0)
          (memcpy-host-to-device rat-reg-2)
          (rat-stitch (i rat) rat-reg-2)
          rat)

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

        (mutate (top rat)
          (let ((candidates (last top (round (* 5/10 (rat-count))))))
            (rat-dissect (i (nth (random (length candidates)) candidates)) rat-reg-0)
            (memcpy-device-to-host rat-reg-0)
            (rat-mutation rat-reg-2 rat-reg-0)
            (memcpy-host-to-device rat-reg-2)
            (rat-stitch (i rat) rat-reg-2)
            rat))

        (randomize (top rat)
          (declare (ignore top))
          (init-fill rat-reg-0)
          (rat-crossover rat-reg-2 rat-reg-0 rat-reg-0)
          (memcpy-host-to-device rat-reg-2)
          (rat-stitch (i rat) rat-reg-2)
          rat)

        (position-rat (rat basements)
          (let ((ok-basement (find-if #'basement-available basements)))
            (push (basement-rats ok-basement) rat)
            (setf (getf rat 'basement) ok-basement
                  (getf rat 'rot) 0.0
                  (getf rat 'x) 0.0
                  (getf rat 'y) 0.0)))

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

        (get-top (rats)
          (sort rats #'(lambda (a b) (< (getf a 'balls) (getf b 'balls)))))

        (update-rats (rats basements)
          (memcpy-device-to-host rat-a-inp)
          (loop for rat in rats do
            (setf (getf rat 'hurt) (mem-aref (+ (* 96 (i rat)) 64)))
            (decf (getf rat 'health) (getf rat 'hurt))
            (incf (getf rat 'balls))
            (setf (getf rat 'x) (mem-aref (+ (* 96 (i rat)) 65)))
            (setf (getf rat 'y) (mem-aref (+ (* 96 (i rat)) 66)))
            (setf (getf rat 'rot) (mem-aref (+ (* 96 (i rat)) 67))))
          (respawn-rats (get-top rats) basements))

        (start ()
          (let ((rats (loop for i from 0 below (rat-count) collect
                        `(health 1.0 hurt 0.0 x 0.0 y 0.0 rot 0.0 balls 0 i ,i)))
                (basements (loop for i from 0 below (ceiling (/ (rat-count) 40)) collect
                              (create-random-basement))))
            nil))







        )

        (start)
        (run-rat)))))

(main)