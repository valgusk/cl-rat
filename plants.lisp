(defun grow-plants (plant-count wall-count wall-blck wall-step wall-start
                                plant-blck plant-step plant-start)
    (with-position-filters plant-count
      ((filter-by-walls wall-count wall-blck wall-step wall-start)
       (filter-by-plants plant-count plant-blck plant-step plant-start))
      (filter-by-walls 0.01 1.0)
      (filter-by-plants 0.01 1.0)
      (filter-by-walls 0.1 -1.0)))

; (defun basements-plants-to-device (basements plant-blk plant-step)
  
;   )