// Pi Calculator using the Chudnovsky Algorithm
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <gmp.h>
#include <unistd.h>
#include <libgen.h>
#include <stdarg.h>
#include <omp.h>
#include <time.h>

// log_2(10)
#define BITS_PER_DIGIT 3.3219280949

// flag for verbose output
int verbose;

// flag for outputting elapsed time
int print_time;

/**
 * Prints a debugging message to stderr only if the the verbose option is given
 * by the user.
 */
void debug(const char* fmt, ...) {
    if (verbose) {
        va_list ap;
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
    }
}

/**
 * Combine two pqr results. Outputs should not be the same as inputs.
 */
void pqr_combine(mpz_t r_p, mpz_t r_q, mpz_t r_r,
    mpz_t pam, mpz_t qam, mpz_t ram,
    mpz_t pmb, mpz_t qmb, mpz_t rmb) {
    mpz_t p, q, r, pam_rmb;
    mpz_inits(p, q, r, pam_rmb, NULL);
    // Pab = Pam * Pmb
    // Qab = Qam * Qmb
    // Rab = Qmb * Ram + Pam * Rmb
    mpz_mul(p, pam, pmb);
    mpz_mul(q, qam, qmb);
    mpz_mul(r, qmb, ram);
    // pam_rmb = pam * rmb
    mpz_mul(pam_rmb, pam, rmb);
    mpz_add(r, r, pam_rmb);
    mpz_set(r_p, p);
    mpz_set(r_q, q);
    mpz_set(r_r, r);
    mpz_clears(p, q, r, pam_rmb, NULL);
}

/**
 * Perform a binary split.
 */
void binary_split(mpz_t r_p, mpz_t r_q, mpz_t r_r, long a, long b) {
    if (b == a + 1) {
        // P(a, a+1) = -(6a-1)(2a-1)(6a-5)
        // pn is the nth term of the expression above (p1 is stored in r_p)
        mpz_t p2, p3;
        mpz_set_ui(r_p, a);
        mpz_mul_ui(r_p, r_p, 6);
        mpz_sub_ui(r_p, r_p, 1);
        mpz_init_set_ui(p2, a);
        mpz_mul_ui(p2, p2, 2);
        mpz_sub_ui(p2, p2, 1);
        mpz_init_set_ui(p3, a);
        mpz_mul_ui(p3, p3, 6);
        mpz_sub_ui(p3, p3, 5);
        mpz_mul(r_p, r_p, p2);
        mpz_mul(r_p, r_p, p3);
        mpz_neg(r_p, r_p);
        mpz_clears(p2, p3, NULL);
        
        // Q(a, a+1) = 10939058860032000a**3
        mpz_set_ui(r_q, a);
        mpz_pow_ui(r_q, r_q, 3);
        mpz_mul_ui(r_q, r_q, 10939058860032000);
        
        // R(a, a+1) = P(a, a+1) * (545140134a + 13591409)
        mpz_set_ui(r_r, a);
        mpz_mul_ui(r_r, r_r, 545140134);
        mpz_add_ui(r_r, r_r, 13591409);
        mpz_mul(r_r, r_r, r_p);
    } else {
        long m = (a + b) / 2;
        mpz_t pam, qam, ram, pmb, qmb, rmb;
        mpz_inits(pam, qam, ram, pmb, qmb, rmb, NULL);
        binary_split(pam, qam, ram, a, m);
        binary_split(pmb, qmb, rmb, m, b);
        pqr_combine(r_p, r_q, r_r, pam, qam, ram, pmb, qmb, rmb);
        mpz_clears(pam, qam, ram, pmb, qmb, rmb, NULL);
    }
}

/**
 * Calculate Pi with `n` iterations and a maximum of `thread` threads.
 */
void chudnovsky(mpf_t r_pi, long n, int threads) {
    // (426880 * sqrt(10005) * Q(1, n)) / (13591409 * Q(1, n) + R(1, n))
    // to retain accuracy, denominator is initially an integer
    mpf_t den, f_q;
    mpf_inits(den, f_q, NULL);
    debug("start binary split\n");

    // multithreaded implementation for binary split
    if (n < threads * 2) threads = 1;
    long interval = n / threads;
    debug("interval: %ld\n", interval);
    int num_vals = threads;
    mpz_t* p_vals = malloc(num_vals * sizeof(mpz_t));
    mpz_t* q_vals = malloc(num_vals * sizeof(mpz_t));
    mpz_t* r_vals = malloc(num_vals * sizeof(mpz_t));

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < num_vals; i++) {
        long start = i * interval + 1;
        long end = (i == num_vals - 1) ? n : (i + 1) * interval + 1;
        mpz_inits(p_vals[i], q_vals[i], r_vals[i], NULL);
        binary_split(p_vals[i], q_vals[i], r_vals[i], start, end);
    }

    debug("performing reduction\n");

    while (num_vals > 1) {
        // reduce in pairs until only one value is left
        int new_num_vals = (num_vals + 1) / 2;
        debug("number of values: current %d, next %d\n", num_vals, new_num_vals);
        mpz_t* new_p_vals = malloc(new_num_vals * sizeof(mpz_t));
        mpz_t* new_q_vals = malloc(new_num_vals * sizeof(mpz_t));
        mpz_t* new_r_vals = malloc(new_num_vals * sizeof(mpz_t));
        
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < num_vals - 1; i += 2) {
            int new_i = i / 2;
            mpz_inits(new_p_vals[new_i], new_q_vals[new_i], new_r_vals[new_i], NULL);
            pqr_combine(new_p_vals[new_i], new_q_vals[new_i], new_r_vals[new_i],
                p_vals[i], q_vals[i], r_vals[i],
                p_vals[i + 1], q_vals[i + 1], r_vals[i + 1]);
            mpz_clears(p_vals[i], q_vals[i], r_vals[i],
                p_vals[i + 1], q_vals[i + 1], r_vals[i + 1], NULL);
        }
        if (num_vals & 1) {
            int new_prev = new_num_vals - 1;
            int prev = num_vals - 1;
            mpz_init_set(new_p_vals[new_prev], p_vals[prev]);
            mpz_init_set(new_q_vals[new_prev], q_vals[prev]);
            mpz_init_set(new_r_vals[new_prev], r_vals[prev]);
            mpz_clears(p_vals[prev], q_vals[prev], r_vals[prev], NULL);
        }
        free(p_vals);
        free(q_vals);
        free(r_vals);
        p_vals = new_p_vals;
        q_vals = new_q_vals;
        r_vals = new_r_vals;
        num_vals = new_num_vals;
    }

    mpz_t p, q, r;
    mpz_inits(p, q, r, NULL);
    mpz_set(p, p_vals[0]);
    mpz_set(q, q_vals[0]);
    mpz_set(r, r_vals[0]);
    mpz_clears(p_vals[0], q_vals[0], r_vals[0], NULL);
    
    debug("combining values\n");
    // numerator = (426880 * sqrt(10005) * q)
    mpf_set_z(f_q, q);
    mpf_sqrt_ui(r_pi, 10005);
    mpf_mul_ui(r_pi, r_pi, 426880);
    mpf_mul(r_pi, r_pi, f_q);
    // denominator = (13591409 * q + r)
    // q used as denominator
    mpz_mul_ui(q, q, 13591409);
    mpz_add(q, q, r);
    mpf_set_z(den, q);
    mpf_div(r_pi, r_pi, den);

    mpz_clears(p, q, r, NULL);
    mpf_clears(den, f_q, NULL);
}

/**
 * Prints program usage to stderr.
 */
static void usage(char* name) {
    fprintf(stderr,
            "Usage: %s [-vqh] [-n THREADS] PREC\n"
                "Options:\n"
                " -v\tshow verbose (debug) messages\n"
                " -q\tdon't print result\n"
                " -t\tprint time taken\n"
                " -n\tspecify number of threads to use\n"
                " -h\tprint help then exit\n"
                " PREC\tnumber of decimal points to calculate (must be non-negative)\n",
            name);
}

int main(int argc, char** argv) {
    long prec = 0;
    int quiet = 0;
    int opt;
    opterr = 0;
    verbose = 0;
    print_time = 0;
    int threads = omp_get_max_threads();
    char* nvalue = 0;
    while ((opt = getopt(argc, argv, "vqhtn:")) != -1) {
        switch (opt) {
            case 'v':
                verbose = 1;
                break;
            case 'q':
                quiet = 1;
                break;
            case 't':
                print_time = 1;
                break;
            case 'n':
                nvalue = optarg;
                break;
            case 'h':
                usage(basename(argv[0]));
                exit(EXIT_SUCCESS);
            case '?':
                if (optopt == 'n') {
                    fprintf(stderr, "Option -%c requires an argument", optopt);
                } else if (isprint(optopt)) {
                    fprintf(stderr, "Unknown option '%c'\n", optopt);
                } else {
                    fprintf(stderr, "Unknown character %x\n", optopt);
                }
            default:
                usage(basename(argv[0]));
                exit(EXIT_FAILURE);
        }
    }
    if (optind == argc) {
        // no positional arguments
        debug("no positional args\n");
        usage(basename(argv[0]));
        exit(EXIT_FAILURE);
    }
    char* err = 0;
    if (nvalue) {
        debug("-n specified\n");
        threads = strtol(nvalue, &err, 10);
        if (*err || threads < 1) {
            usage(basename(argv[0]));
            exit(EXIT_FAILURE);    
        }
    }
    prec = strtol(argv[optind], &err, 10);
    if (*err || prec < 0) {
        usage(basename(argv[0]));
        exit(EXIT_FAILURE);
    }
    debug("precision: %d\n", prec);
    debug("max number of threads: %d\n", threads);

    long prec_bits = (prec + 2) * BITS_PER_DIGIT;
    mpf_set_default_prec(prec_bits);
    mpf_t pi;
    mpf_init(pi);
    int n = prec / 14 > 1 ? prec / 14 : 2;
    clock_t start = clock();
    chudnovsky(pi, n, threads);
    double duration = (double)(clock() - start) / CLOCKS_PER_SEC;
    if (print_time) {
        printf("%lf\n", duration);
    }
    if (!quiet) {
        gmp_printf("%.*Ff\n", prec, pi);
    }
    mpf_clear(pi);
    return 0;
}
