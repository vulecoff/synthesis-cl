
    !(foldl-atom (0 1 2 3 4) 0 $acc $f 
        (let $x $f (+ 1 $acc))
    ) ;; Foldl-atom bug in both hyperon and mettalog


!(let $ls (1 2 3 4) 
    (foldl-atom $ls 0 $acc $f 
        (let () () (+ 1 $acc))
    )
)


!(foldl-atom (0 1 2 3 4) 0 $acc $f 
    (let* (
        ; ($x $f)
        (() (println! $f))
    )(none))
)


!(let $x 2
    (let $y $x 
        (let $y 3 
            (result $x $y)
        )
    )
)