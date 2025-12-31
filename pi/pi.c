/* Pi Calculator using the binary splitting method for the Chudnovsky Algorithm
 * from https://en.wikipedia.org/wiki/Chudnovsky_algorithm
*/

#include <omp.h>
#include <gmp.h>
#include <stdlib.h>
#include <stdio.h>

#define BITS_PER_DIGIT 3.32192809489

void binary_split(mpz_t r_Pab, mpz_t r_Qab, mpz_t r_Rab, long a, long b)
{
    // P(a, a+1) = -(6a-1)(2a-1)(6a-5)
    // Q(a, a+1) = 10939058860032000a**3
    // R(a, a+1) = P(a, a+1) * (545140134a + 13591409)
    mpz_t Pab, Qab, Rab;
    mpz_inits(Pab, Qab, Rab, NULL);
    // printf("binary split inside a=%ld b=%ld\n", a, b);
    
    if (b == a + 1)
    {
        // calculate P. initialize 3 variables for the terms
        mpz_t P1, P2, P3;
        
        mpz_init_set_si(P1, a);
        mpz_mul_ui(P1, P1, 6);
        mpz_sub_ui(P1, P1, 1);
        mpz_neg(P1, P1);
        
        mpz_init_set_si(P2, a);
        mpz_mul_ui(P2, P2, 2);
        mpz_sub_ui(P2, P2, 1);
        
        mpz_init_set_si(P3, a);
        mpz_mul_ui(P3, P3, 6);
        mpz_sub_ui(P3, P3, 5);
        
        //mpz_set(Pab, P1);
        mpz_mul(Pab, P1, P2);
        mpz_mul(Pab, Pab, P3);
        
        // calculate Q
        mpz_set_si(Qab, a);
        mpz_pow_ui(Qab, Qab, 3);
        mpz_mul_ui(Qab, Qab, 10939058860032000);
        
        // calculate R
        mpz_set_si(Rab, a);
        mpz_mul_ui(Rab, Rab, 545140134);
        mpz_add_ui(Rab, Rab, 13591409);
        mpz_mul(Rab, Rab, Pab);
        
        mpz_clears(P1, P2, P3, NULL);
    }
    else
    {
        long m = (a + b) / 2;
        mpz_t Pam, Qam, Ram, Pmb, Qmb, Rmb, Pam_Rmb;
        mpz_inits(Pam, Qam, Ram, Pmb, Qmb, Rmb, Pam_Rmb, NULL);
        
        binary_split(Pam, Qam, Ram, a, m);
        binary_split(Pmb, Qmb, Rmb, m, b);
        
        mpz_mul(Pab, Pam, Pmb);
        mpz_mul(Qab, Qam, Qmb);
        mpz_mul(Rab, Qmb, Ram);
        // use Pam_Rmb to calculate second term Pam * Rmb
        mpz_mul(Pam_Rmb, Pam, Rmb);
        mpz_add(Rab, Rab, Pam_Rmb);
        
        mpz_clears(Pam, Qam, Ram, Pmb, Qmb, Rmb, Pam_Rmb, NULL);
    }

    mpz_set(r_Pab, Pab);
    mpz_set(r_Qab, Qab);
    mpz_set(r_Rab, Rab);
    mpz_clears(Pab, Qab, Rab, NULL);
    return;
}

void chudnovsky(mpf_t r_pi, long n, int threads, unsigned long prec_bits)
{
    // (426880 * sqrt(10005) * Q(1, n)) / (13591409 * Q(1, n) + R(1, n))
    // to retain accuracy, denominator is initially an integer
    mpf_set_default_prec(prec_bits);

    // mpz_t P[threads];
    // mpz_t Q[threads];
    // mpz_t R[threads];
    // for (int i = 0; i < threads; i++)
    // {
    //     mpz_inits(P[i], Q[i], R[i], NULL);
    // }

    mpz_t P, Q, R;
    mpz_inits(P, Q, R, NULL);

    mpz_t int_den;
    mpz_init(int_den);
    mpf_t den, f_Qab;
    mpf_inits(den, f_Qab, NULL);
    
    // multiprocessing implementation
    long interval = n / threads;
    fprintf(stderr, "n %ld\nthreads %d\n", n, threads);
    fprintf(stderr, "interval %ld\n", interval);
    
    #pragma omp parallel for num_threads(threads) ordered
    for (int i = 0; i < threads - 1; i++)
    {
        long start = i * interval + 1;
        long end = (i + 1) * interval + 1;
        mpz_t Pab, Qab, Rab;
        mpz_inits(Pab, Qab, Rab, NULL);
        binary_split(Pab, Qab, Rab, start, end);

        #pragma omp ordered
        {
            // if first iteration, set P, Q, R to results
            if (i == 0)
            {
                mpz_set(P, Pab);
                mpz_set(Q, Qab);
                mpz_set(R, Rab);
            }
            else 
            {
                mpz_t Pab_Rmb;
                mpz_init(Pab_Rmb);
                mpz_mul(Pab_Rmb, P, Rab);
                mpz_mul(R, Qab, R);
                mpz_add(R, R, Pab_Rmb);
                
                mpz_mul(P, P, Pab);
                mpz_mul(Q, Q, Qab);
                mpz_clear(Pab_Rmb);
            }
        }
        mpz_clears(Pab, Qab, Rab, NULL);
    }
    mpz_t Pab, Qab, Rab;
    mpz_inits(Pab, Qab, Rab, NULL);
    binary_split(Pab, Qab, Rab, (threads - 1) * interval + 1, n);
    mpz_t Pab_Rmb;
    mpz_init(Pab_Rmb);
    mpz_mul(Pab_Rmb, P, Rab);
    mpz_mul(R, Qab, R);
    mpz_add(R, R, Pab_Rmb);
    mpz_mul(P, P, Pab);
    mpz_mul(Q, Q, Qab);
    mpz_clear(Pab_Rmb);
    mpz_clears(Pab, Qab, Rab, NULL);
    #pragma omp barrier
    
    mpf_set_z(f_Qab, Q);
    
    fprintf(stderr, "calculating numerator\n");
    mpf_set_d(r_pi, 10005.0);
    mpf_sqrt(r_pi, r_pi);
    mpf_mul_ui(r_pi, r_pi, 426880);
    mpf_mul(r_pi, r_pi, f_Qab);
    
    fprintf(stderr, "calculating denominator\n");
    mpz_mul_ui(int_den, Q, 13591409);
    mpz_add(int_den, int_den, R);
    mpf_set_z(den, int_den);
    mpz_clear(int_den);
    
    mpf_div(r_pi, r_pi, den);

    mpz_clears(P, Q, R, NULL);
    mpf_clears(den, f_Qab, NULL);
    return;
}

int main(int argc, char* argv[])
{
    // unsigned long prec = atol(argv[1]);
    unsigned long prec = 20;
    unsigned long prec_bits = (prec + 2) * BITS_PER_DIGIT + 3;
    fprintf(stderr, "precision: %ld\n", prec);
    fprintf(stderr, "prec_bits: %ld\n", prec_bits);
    mpf_set_default_prec(prec_bits);
    mpf_t pi;
    mpf_init(pi);
    int threads = omp_get_max_threads();
    int n = (prec / 14 > 1) ? prec / 14 + 1 : 2;
    if (n < threads)
    {
        threads = n / 2;
    }
    chudnovsky(pi, n, threads, prec_bits);
    gmp_printf("%.*Ff\n", prec, pi);
    
    return 0;
}
