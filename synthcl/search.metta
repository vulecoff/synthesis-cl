;; 

;; generate all possible programs with a given set of combinators

!(import! &self lib)
!(import! &self cl)

;; (program $depth $prog)

;; add & return progn if not exists
(= (add-progn-if-nexists! $depth $progn)
    (if (== (collapse (match &self (program $depth $progn) exists)) ())    
        (let () (add-atom &self (program $depth $progn))
            $progn
        )
        (empty)
    )   
)

;; generate all programs up to a certain depth
;; non-deterministic semantics make the logic very elegant
(= (generate! $d $combinators)
    (if (> $d 1)
        (let* (
            ((quote $last-progn) (generate! (- $d 1) $combinators))
            ($cur (superpose $combinators))
            ($nxt-prog-1 (quote ($cur $last-progn)))
            ($nxt-prog-2 (quote ($last-progn $cur)))
            ; (() (println! (at $d $nxt-prog-1 $nxt-prog-2)))
            
        ) (superpose (
            (add-progn-if-nexists! $d $nxt-prog-1)
            (add-progn-if-nexists! $d $nxt-prog-2)
        )))
        (let $p (superpose $combinators)
            (add-progn-if-nexists! 1 (quote $p))
        )
    )
)

(= (get-repr $s $list)
    (let* (
        ($pair-ls (filter-atom $list $x (== (car-atom $x) $s)))
        ($pair (car-atom $pair-ls))
        ) (index-atom $pair 1)) ;; TODO: catch/throw error if not found
)

(= (try-reduce $fact)
    (let 
        (((quote $a) (quote $b)) (quote $c)) $fact
        (if (== ($a $b) $c)
            True
            False
        ) 
    )
)

;; ((a x b) rhs) 
;; for each fact/rule, assign symbol=get-repr, try reduce 
(= (try-assignments $repr-list $facts-list)
    (let ($fact $rest) (decons-atom $facts-list)
        (if (== $fact ())
            True 
            (let* (
               ( (($a $b) $c) $fact )
               ($x (get-repr $a $repr-list))
               ($y (get-repr $b $repr-list))
               ($z (get-repr $c $repr-list))
            ;   ( (($x $y) $z) (( (get-repr $a $repr-list) (get-repr $b $repr-list)) (get-repr $c $repr-list))  )
              ($reducible (try-reduce (($x $y) $z)))
            )
                (and $reducible (try-assignments $repr-list $rest))
            )
        )
    )
)

(= (brute-search $d $symbs-list $facts-list)
    (let* (
        ($repr-list (map-atom $symbs-list $x (match &self (program $d $y) ($x $y))))
        ($fit (try-assignments $repr-list $facts-list))
        ; (() (println! $repr-list))
    )
        (if $fit $repr-list (empty))
    )
)

!(generate! 5 (S K))
!(brute-search 5 (t f) (((t f) t) ((t t) t)))
;; try-assignments reprl facts ;;
    ;; rerpr: (neg t f)  --> (neg (quote not)) (t (quote True)) (f (quote False))
    ;; ((neg t) f) ((neg f) t)

; !(try-assignments ( (neg (quote not)) (t (quote True)) (f (quote False))) (((neg t) f) ((neg f) t)))
