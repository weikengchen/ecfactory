#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <NTL/ZZ.h>
#include <NTL/tools.h>
#include <omp.h>

using namespace std;
using namespace NTL;

#define SUBGROUP_SECURE
#define MAX_Q_BITS 384
#define LOGFILE "./LOG.txt"

/*
  This code modifies [Algorithm 1, BN06] to search for candidate parameters of
  Barreto-Naehrig curves with high 2-adicity. (The curve itself can be explicitly
  constructed by following the second half of [Algorithm 1, BN06].)

  This code was used to find the BN curve in [BCTV13].

  [BCTV13] = "Succinct Non-Interactive Arguments for a von Neumann Architecture"
  [BN06]   = "Pairing-Friendly Elliptic Curves of Prime Order"
*/

int main(int argc, char **argv) {
	if (argc != 4) {
		cout << "usage: " << argv[0] << " [rand_seed even_offset wanted_two_adicity]\n";
		return 0;
	}

	/* Collect inputs */
	ZZ seed;
	conv(seed, atoi(argv[1]));
	long even_offset = atoi(argv[2]);
	long wanted_two_adicity = atoi(argv[3]);

	/* |x| ~ 64 bits so that |q| = 4 * |x| ~ 256 bits */
	long num_x_bits = 95;
	long cur_lowest_hw = -1;

	int num_threads = omp_get_max_threads();
	ZZ random_x[num_threads * 10000];

	SetSeed(seed);
	while (1) {
		/* Sample x */
		for(int i = 0; i < num_threads; i++) {
			for(int j = 0; j < 10000; j++) {
				random_x[i * 10000 + j] = RandomLen_ZZ(num_x_bits);
			}
		}

		#pragma omp parallel for default(none) shared(cur_lowest_hw)
		for(int i = 0; i < num_threads; i++) {
			ZZ* cur_start = &random_x[i * 10000];

			for(int j = 0; j < 10000; j++) {
				/**
				 * Make x even and divisible by a large power of 2 to improve two adicity.
				 * The resulting q is s.t. -1 is a square in Fq.
				 */
				ZZ x = cur_start[j];

				x = ((x >> even_offset) << even_offset);

				/* Uncomment to make x odd and ensure that -1 is a nonsquare in Fq. */
				SetBit(x, 0);

				int hw = 0;
				for (int i = 0; i < NumBits(x); i++) {
					hw += bit(x, i);
				}

				if (cur_lowest_hw != -1 && hw > cur_lowest_hw) {
					continue;
				}

				/**
				 * Compute candidate BN parameters using the formulas:
				 * t = 6*x^2 + 1,
				 * q = 36*x^4 + 36*x^3 + 24*x^2 + 6*x + 1
				 * r = q - t + 1
				 * (see [BN06])
				 */
				ZZ x2 = x * x;
				ZZ x3 = x2 * x;
				ZZ x4 = x3 * x;
				ZZ t = 6 * x2 + 1;
				ZZ q = 36 * x4 + 36 * x3 + 24 * x2 + 6 * x + 1;
				ZZ r = q - t + 1;

				long num_q_bits = NumBits(q);
				long two_adicity = NumTwos(r - 1);

				if (num_q_bits > MAX_Q_BITS) {
					continue;
				}

				#ifdef SUBGROUP_SECURE
				ZZ h2 = 36 * x4 + 36 * x3 + 30 * x2 + 6 * x + 1;
				if (!ProbPrime(h2)) {
					continue;
				}
				ZZ q2 = q * q;
				ZZ q4 = q2 * q2;
				ZZ h3 = (q4 - q2 + 1) / r;
				if (!ProbPrime(h3)) {
					continue;
				}
				#endif

				if (ProbPrime(r) && ProbPrime(q) && (two_adicity >= wanted_two_adicity)) {
					if (cur_lowest_hw != -1 && hw > cur_lowest_hw) {
						continue;
					}
					#pragma omp critical (print_and_update)
					{
						cout << "x = " << x << "\n";
						cout << "q = " << q << "\n";
						cout << "r = " << r << "\n";
						cout << "hw(x) = " << hw << "\n";
						cout << "log2(q) =" << num_q_bits << "\n";
						cout << "ord_2(r-1) = " << two_adicity << "\n";
						cur_lowest_hw = hw;
						cout.flush();

						#ifdef LOGFILE
						std::filebuf fb;
						fb.open(LOGFILE, std::ios::out | std::ios::app);

						std::ostream os(&fb);
						os << "x = " << x << "\n";
						os << "q = " << q << "\n";
						os << "r = " << r << "\n";
						os << "hw(x) = " << hw << "\n";
						os << "log2(q) =" << num_q_bits << "\n";
						os << "ord_2(r-1) = " << two_adicity << "\n";
						os << "\n";
						fb.close();
						#endif
					}
				}
			}
		}
	}
}
