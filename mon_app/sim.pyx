# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time as c_time
cimport cython

cdef int N_LEVELS = 14
cdef int MAX_K = 6

cdef double BUYIN_REQ[14]
cdef int    LCOUNT[14]
cdef double LMULT[14][6]
cdef double LCUM[14][6]

@cython.boundscheck(False)
@cython.wraparound(False)
def load_levels(dict levels):
    cdef int lvl, i, j, k
    cdef dict data, mp
    cdef list items
    cdef double s, p

    for lvl in range(1, N_LEVELS+1):
        i = lvl - 1
        data = <dict> levels[lvl]
        BUYIN_REQ[i] = float(data["buy_in_required"])
        mp = <dict> data["multiplicateurs"]
        items = sorted(mp.items(), key=lambda kv: float(kv[0]))
        k = len(items)
        if k > MAX_K:
            raise ValueError(f"Max {MAX_K} multiplicateurs par niveau (niveau {lvl})")
        LCOUNT[i] = k

        s = 0.0
        for j in range(k):
            LMULT[i][j] = float(items[j][0])
            p = float(items[j][1])
            s += p
            LCUM[i][j] = s

        if s != 0.0:
            for j in range(k):
                LCUM[i][j] /= s

@cython.cfunc
cdef inline int rand_int(int n) nogil:
    return <int>((rand() / (<double>RAND_MAX + 1.0)) * n)

@cython.cfunc
cdef inline double rand_u() nogil:
    return rand() / (<double>RAND_MAX + 1.0)

@cython.cfunc
cdef inline int find_level(double prime) nogil:
    cdef double p10 = prime / 10.0
    cdef int idx
    for idx in range(N_LEVELS-1, -1, -1):
        if p10 >= BUYIN_REQ[idx]:
            return idx
    return 0

@cython.cfunc
cdef inline double draw_mult(int lvl_idx) nogil:
    cdef int k = LCOUNT[lvl_idx]
    cdef double u = rand_u()
    cdef int j
    for j in range(k):
        if u <= LCUM[lvl_idx][j]:
            return LMULT[lvl_idx][j]
    return LMULT[lvl_idx][k-1]

@cython.boundscheck(False)
@cython.wraparound(False)
def simuler_tournoi_cython(int n_joueurs,
                           bint space_ko,
                           double stack_initial,
                           double prime_initiale,
                           unsigned int seed=0):
    cdef int i
    if seed == 0:
        srand(<unsigned int>c_time(NULL))
    else:
        srand(seed)

    cdef list prime = [prime_initiale] * n_joueurs
    cdef list gain  = [0.0] * n_joueurs
    cdef list stack = [stack_initial] * n_joueurs
    cdef list alive = list(range(n_joueurs))
    cdef list place = [0] * n_joueurs

    cdef int alive_n = n_joueurs
    cdef int k1, k2, i1, i2, loser_pos
    cdef int current_place = n_joueurs
    cdef double mise, r, mult, g, tmp
    cdef int gagnant_idx, perdant_idx
    cdef double total_gains = 0.0

    while alive_n > 1:
        k1 = rand_int(alive_n)
        k2 = rand_int(alive_n)
        while k2 == k1:
            k2 = rand_int(alive_n)

        i1 = <int> alive[k1]
        i2 = <int> alive[k2]

        r = rand_u()
        if r < 0.5:
            gagnant_idx = i1; perdant_idx = i2; loser_pos = k2
        else:
            gagnant_idx = i2; perdant_idx = i1; loser_pos = k1

        mise = stack[gagnant_idx] if stack[gagnant_idx] < stack[perdant_idx] else stack[perdant_idx]
        stack[gagnant_idx] += mise
        stack[perdant_idx] -= mise

        if stack[perdant_idx] <= 0.0:
            if space_ko:
                mult = draw_mult(find_level(prime[perdant_idx]))
            else:
                mult = 1.0

            g = prime[perdant_idx] * mult
            tmp = 0.5 * g
            gain[gagnant_idx] += tmp
            prime[gagnant_idx] += tmp
            total_gains += tmp

            place[perdant_idx] = current_place
            current_place -= 1
            alive[loser_pos] = alive[alive_n - 1]
            alive_n -= 1

    i1 = <int> alive[0]
    g = prime[i1]
    gain[i1] += g
    total_gains += g
    place[i1] = 1

    cdef list joueurs = []
    cdef dict dct
    for i in range(n_joueurs):
        dct = {"prime": float(prime[i]), "gain": float(gain[i]),
               "stack": float(stack[i]), "place": int(place[i])}
        joueurs.append(dct)
    return joueurs, float(total_gains)
