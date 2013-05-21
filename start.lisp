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

        (give-x-y (target basement &optional (x (random 1.0)) (y (random 1.0)) (thld 0.01))
          (if (find-if #'(lambda (object) (and (> thld (abs (- (getf object 'x) x)))
                                               (> thld (abs (- (getf object 'y) y)))
                                               (incf x (* (random 2.0) (- (getf object 'x) x)))
                                               (incf x (* (random 2.0) (- (getf object 'x) x)))))
                       (append (basement-rats basement)
                               (basement-cats basement)
                               (basement-plants basement)
                               (basement-walls basement)))
              (progn
                (incf x (- (random 0.1) 0.05))
                (incf y (- (random 0.1) 0.05))
                (give-x-y target basement (max 1.0 (min 0.0 x)) (max 1.0 (min 0.0 y)) thld))
              (setf (getf target 'x) x
                    (getf target 'y) y)))

        (position-rat (rat basements)
          (let ((ok-basement (find-if #'basement-available basements)))
            (push (basement-rats ok-basement) rat)
            (setf (getf rat 'basement) ok-basement)
            (give-x-y rat ok-basement)
            (setf (getf rat 'rot) 0.0)))

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
                        `(balls 0 health 1.0 hurt 0.0 x 0.0 y 0.0 rot 0.0 i ,i))))))







        )))))

(main)