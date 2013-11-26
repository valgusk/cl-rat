(defun call-cats (cat-count wall-count wall-blck wall-step wall-start)
  (with-memory-block (candidates-blk 'int (* 100 100))
      (flet ((filter (dist sign)
               (cu-free-from
                 candidates-blk dist sign
                 wall-blck wall-step wall-count wall-start
                 :grid-dim (list (ceiling (/ (* 100 100 wall-count) 256)) 1 1)
                 :block-dim '(256 1 1))))
        (cu-clear-candidates candidates-blk)
        (filter 0.03 1.0) ;evade walls
        (memcpy-device-to-host candidates-blk)
        (loop for i below 10000
              for x-y = (list (floor (/ i 100)) (rem i 100))
              when (plusp (mem-aref candidates-blk i))
              collect x-y into cats
              finally (return (nrandom-pick cat-count cats))))))
