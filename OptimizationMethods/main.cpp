#include <iostream>
#include <format>

typedef long long ll;
typedef long double ld;

using namespace std;

ld compute(ld x1, ld x2) {
    return x1 * x1 * x1 - x1 * x2 + x2 * x2 - 2 * x1 + 3 * x2 - 4;
}

ld computeDerivative(ld x1, ld x2, ll i) {
    if (i & 1) // Нечёт шаг - шаг по x1
        return 3 * x1 * x1 - x2 - 2;
    return -x1 + 2 * x2 + 3;
}

int main() {
    ld eps = 0.1;
    ld dx = 0;
    ld t = 1;
    ld x1 = 0, x2 = 0;

    ld fprev;
    ld fcur = compute(x1, x2);

    ld *x1p = &x1;
    ld *x2p = &x2;

    cout << format("{:<3} {:^20} {:^20} {:^7} {:^7} {:^7} {:^7} {:^20} {:^7}", "i ", "xp", "xm", "df", "dx", "f(xp)", "f(xm)", "x", "f(x)") << endl;
    ll i = 1;
    do {
        fprev = fcur;

        ld df = computeDerivative(x1, x2, i);
        dx = abs(t * df);

        *x1p += dx;
        pair<ld, ld> x1x2p = {*x1p, *x2p};
        ld fp = compute(x1, x2);

        *x1p -= 2 * dx;
        pair<ld, ld> x1x2m = {*x1p, *x2p};
        ld fm = compute(x1, x2);

        fcur = fm;
        if (fp < fcur) {
            *x1p += 2 * dx;
            fcur = fp;
        }

        swap(x1p, x2p);

        cout << format("{:<3} {:^20} {:^20} {:^7.2f} {:^7.2f} {:^7.2f} {:^7.2f} {:^20} {:^7.2f}",
                       i++, format("({:<5.2f}; {:<5.2f})", x1x2p.first, x1x2p.second), format("({:<5.2f}; {:<5.2f})", x1x2m.first, x1x2m.second), df,dx, fp, fm, format("({:<5.2f}; {:<5.2f})",x1, x2), fcur) << endl;
    } while (abs(fcur - fprev) > eps);

    return 0;
}
