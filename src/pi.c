// Pi Calculator using the Chudnovsky Algorithm
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <gmp.h>
#include <unistd.h>
#include <libgen.h>
#include <stdarg.h>

// log_2(10)
#define BITS_PER_DIGIT 3.3219280949

int verbose;

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
        
        // Pab = Pam * Pmb
        // Qab = Qam * Qmb
        // Rab = Qmb * Ram + Pam * Rmb
        mpz_mul(r_p, pam, pmb);
        mpz_mul(r_q, qam, qmb);
        mpz_mul(r_r, qmb, ram);
        // also, pam = pam * rmb
        mpz_mul(pam, pam, rmb);
        mpz_add(r_r, r_r, pam);
        mpz_clears(pam, qam, ram, pmb, qmb, rmb, NULL);
    }
}

void chudnovsky(mpf_t r_pi, long n, int nthreads) {
    // (426880 * sqrt(10005) * Q(1, n)) / (13591409 * Q(1, n) + R(1, n))
    // to retain accuracy, denominator is initially an integer
    mpz_t p, q, r;
    mpf_t den, f_q;
    mpz_inits(p, q, r, NULL);
    mpf_inits(den, f_q, NULL);
    debug("start binary split\n");
    binary_split(p, q, r, 1, n + 1);
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

static void usage(char* name) {
    fprintf(stderr,
            "Usage: %s [-vqh] PREC\n"
                "Options:\n"
                " -v\tshow verbose (debug) messages\n"
                " -q\tdon't print result\n"
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
    while ((opt = getopt(argc, argv, "vqh")) != -1) {
        switch (opt) {
            case 'v':
                verbose = 1;
                break;
            case 'q':
                quiet = 1;
                break;
            case 'h':
                usage(basename(argv[0]));
                exit(EXIT_SUCCESS);
            case '?':
                if (isprint(optopt)) {
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
        debug("No positional args\n");
        usage(basename(argv[0]));
        exit(EXIT_FAILURE);
    }
    char* err = 0;
    prec = strtol(argv[optind], &err, 10);
    if (*err || prec < 0) {
        usage(basename(argv[0]));
        exit(EXIT_FAILURE);
    }
    debug("precision: %d\n", prec);

    long prec_bits = (prec + 2) * BITS_PER_DIGIT;
    mpf_set_default_prec(prec_bits);
    mpf_t pi;
    mpf_init(pi);
    int n = prec / 14 > 1 ? prec / 14 + 1 : 2;
    chudnovsky(pi, n, 1);
    if (!quiet) {
        gmp_printf("%.*Ff\n", prec, pi);
    }
    mpf_clear(pi);
    return 0;
}
