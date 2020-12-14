#include <array>
#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <chrono>
#include <functional>

# define N 200

typedef double (*routine_type)(double, double, int);

double f(double x) {
	double result = std::pow(sin(1 / x), 2) / std::pow(x, 2);
	return result;
}

double trapezoid_routine(double a, double b, int n) {
	double h = (b - a) / n;
	double sum_p = 0;
	double result;

	for (int i = 1; i < n - 1; i++) {
		sum_p += f(a + h * i);
	}

	result = h * ((f(a) + f(b)) / 2 + sum_p);

	return result;
}

double trapezoid_routine_atomic(double a, double b, int n) {
	double h = (b - a) / n;
	double sum_p = 0, f_i;
	double result;

	#pragma omp parallel for
	for (int i = 1; i < n - 1; i++) {
		f_i = f(a + h * i);
		#pragma omp critical
		sum_p += f_i;
	}

	result = h * ((f(a) + f(b)) / 2 + sum_p);

	return result;
}

double trapezoid_routine_critical_section(double a, double b, int n) {
	double h = (b - a) / n;
	double sum_p = 0, f_i;
	double result;

	#pragma omp parallel for
	for (int i = 1; i < n - 1; i++) {
		f_i = f(a + h * i);
		#pragma omp critical
		sum_p += f_i;
	}

	result = h * ((f(a) + f(b)) / 2 + sum_p);

	return result;
}

double trapezoid_routine_locks(double a, double b, int n) {
	double h = (b - a) / n;
	double sum_p = 0, f_i;
	double result;
	omp_lock_t lock;

	omp_init_lock(&lock);
	#pragma omp parallel for
	for (int i = 1; i < n - 1; i++) {
		f_i = f(a + h * i);
		omp_set_lock(&lock);
		sum_p += f_i;
		omp_unset_lock(&lock);
	}
	omp_destroy_lock(&lock);

	result = h * ((f(a) + f(b)) / 2 + sum_p);

	return result;
}

double trapezoid_routine_reduction(double a, double b, int n) {
	double h = (b - a) / n;
	double sum_p = 0, f_i;
	double result;

	#pragma omp parallel for reduction(+:sum_p)
	for (int i = 1; i < n - 1; i++) {
		f_i = f(a + h * i);
		sum_p += f_i;
	}

	result = h * ((f(a) + f(b)) / 2 + sum_p);

	return result;
}

double trapezoid(double a, double b, double eps = 0.001, std::function<double(double, double, int)> routine = trapezoid_routine, int n = 10) {
	double I_1 = trapezoid_routine(a, b, n), I_2 = trapezoid_routine(a, b, 2 * n);

	while (std::abs(I_1 - I_2) > eps * std::abs(I_2)) {
		n *= 2;
		I_1 = I_2;
		I_2 = trapezoid_routine(a, b, 2 * n);
	}

	return I_2;
}

void time_complex(double eps, int threads_cnt, double a, double b, std::function<double(double, double, int)> routine, std::string r_name) {
	std::chrono::steady_clock::time_point start, end;
	int time_sum = 0;

	for (int i = 0; i < N; i++) {
		start = std::chrono::steady_clock::now();
		trapezoid(a, b, eps, routine);
		end = std::chrono::steady_clock::now();
		time_sum = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	}
	std::cout << r_name << ": " << time_sum / N << " ns" << std::endl;
}

int main() {
	double eps = 0.000001;
	std::array<int, 3> threads_cnt = {4, 8, 16};
	double a = 10, b = 100;

	std::cout << "eps: " << eps << "; [a, b]: [" << a << ", " << b << "]" << std::endl;

	for (int i = 0; i < threads_cnt.size(); i++) {
		omp_set_num_threads(threads_cnt[i]);
		std::cout << "threads count: " << threads_cnt[i] << std::endl;

		time_complex(eps, threads_cnt[i], a, b, trapezoid_routine, "trapezoid_routine");
		time_complex(eps, threads_cnt[i], a, b, trapezoid_routine_atomic, "trapezoid_routine_atomic");
		time_complex(eps, threads_cnt[i], a, b, trapezoid_routine_critical_section, "trapezoid_routine_critical_section");
		time_complex(eps, threads_cnt[i], a, b, trapezoid_routine_locks, "trapezoid_routine_locks");
		time_complex(eps, threads_cnt[i], a, b, trapezoid_routine_reduction, "trapezoid_routine_reduction");
	}
}
