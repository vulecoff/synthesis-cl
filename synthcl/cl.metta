!(import! &self lib)

(: K (-> Atom Atom))
(: K (-> Atom Atom Atom))
(: S (-> Atom Atom Atom Atom))
(: S (-> Atom Atom Atom))
(: S (-> Atom Atom))
(= (K $x $y) $x)
; (= ((K $x) $y) (K $x $y))
(= (S $x $y $z) ( ($x $z) ($y $z)))
; (= ((S $x) $y) (S $x $y))
; (= ((S $x $y) $z) (S $x $y $z))

!(make-uncurry! &self K 1 2)
!(make-uncurry! &self S 1 3)

; (is true (K K))
; (is false K)
; (is & ((S (S (S S))) (K (K K))))
; (is v ((S S) (K (K K))))
; (is ~ ((S ((S K) S)) (K K)))

; (= (equiv $x)
;     (match &self
;         (is $y $x)
;         $y
;     )
; )
; (= (equiv $x)
;     (match &self
;         (is $x $y)
;         $y
;     )
; )

; (= (get-symb $x) 
;     (match &self (is $x $y) $y)
; )

; (= (evaluate $opr $x $y) 
;     (((get-symb $opr) (get-symb $x)) (get-symb $y))
; )
; (= (evaluate $opr $x)
;     ((get-symb $opr) (get-symb $x))
; )

; !(evaluate & true false)
; !(evaluate ~ false)