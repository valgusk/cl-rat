(defun grow-plants (plant-count wall-count wall-blck wall-step wall-start)
  (with-memory-block (candidates-blk 'int (* 100 100))
      (flet ((filter-by-walls (dist sign)
               (cu-free-from
                 candidates-blk dist sign
                 wall-blck wall-step wall-count wall-start
                 :grid-dim (list (ceiling (/ (* 100 100 wall-count) 256)) 1 1)
                 :block-dim '(256 1 1))))
        (cu-clear-candidates candidates-blk)
        (filter-by-walls 0.01 1.0) ;cannot grow on wall
        (filter-by-walls 0.1 -1.0) ;should have a wall nearby
        (memcpy-device-to-host candidates-blk)
        (loop for i below 10000
              for x-y = (list (floor (/ i 100)) (rem i 100))
              when (plusp (mem-aref candidates-blk i))
              collect x-y into plants
              finally (return (nrandom-pick plant-count plants))))))