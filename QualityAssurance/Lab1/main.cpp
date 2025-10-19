#include <iostream>
#include <vector>
#include <algorithm>

typedef long long ll;
typedef long double ld;

#define d_c(type, var); type var; cin >> var;
#define d_cl(var); d_c(ll, var)
#define d_cd(var); d_c(ld, var)
#define d_cs(var); d_c(string, var)

using namespace std;

ld getCatchUpTime(pair<ll, ll> vt1, pair<ll, ll> vt2) {
    return ld(1.0) * abs(vt2.first - vt1.first) / abs(vt1.second - vt2.second);
}

ld calcDist(pair<ld, ll> velo, ld t) {
    return velo.first + velo.second * t;
}

ld getDist(pair<ll, ll> vt1, pair<ll, ll> vt2, ld t) {
    return abs(vt1.first - vt2.first + (vt1.second - vt2.second) * t);
}

int main() {
    d_cl(n);
    vector<pair<ld, ll>> velo(n);

    for (int i = 0; i < n; i++) { d_cl(x);d_cl(v);
        velo[i] = {x, v};
    }

    sort(velo.begin(), velo.end(),
         [](const pair<ll, ll> &velo1, const pair<ll, ll> &velo2) { return velo1.first < velo2.first; });

    if (velo[0].second <= velo[n - 1].second) {
        cout << 0 << ' ' << velo[n - 1].first - velo[0].first << endl;
        return 0;
    }

    // время, индекс
    vector<pair<ld, ll>> leader_change_t = {{0, n - 1}};

    pair<ld, ll> max_leader = velo[n - 1];
    ld max_leader_t = 0;
    for (ll i = n - 2; i >= 0; i--) {
        if (velo[i].second <= max_leader.second) {
            continue;
        }

        ld cur_leader_t = getCatchUpTime({calcDist(velo[i], max_leader_t), velo[i].second}, max_leader);
        leader_change_t.emplace_back(cur_leader_t,i);

        if (max_leader_t < cur_leader_t) {
            max_leader = {calcDist(velo[i], cur_leader_t), velo[i].second};
            max_leader_t = cur_leader_t;
        }
    };

    // время, индекс
    vector<pair<ld, ll>> looser_change_t = {{0, 0}};

    pair<ld, ll> max_looser = velo[0];
    ld max_looser_t = 0;
    for (ll i = 1; i < n; i++) {
        if (velo[i].second >= max_looser.second) {
            continue;
        }

        ld cur_looser_t = getCatchUpTime(max_looser, {calcDist(velo[i], max_looser_t), velo[i].second});
        looser_change_t.emplace_back(cur_looser_t, i);

        if (max_looser_t < cur_looser_t) {
            max_looser = {calcDist(velo[i], cur_looser_t), velo[i].second};
            max_looser_t = cur_looser_t;
        }
    }

    ll leader_ind = 0;
    ll looser_ind = 0;

    sort(leader_change_t.begin(), leader_change_t.end(), [](const pair<ld, ll>& l1, const pair<ld, ll>& l2) {return l1.first < l2.first;});
    sort(looser_change_t.begin(), looser_change_t.end(), [](const pair<ld, ll>& l1, const pair<ld, ll>& l2) {return l1.first < l2.first;});

    pair<ld, ld> minTimeAndDist = {
            0,
            getDist(velo[0], velo[n - 1], 0)
    };

    while (leader_ind < leader_change_t.size() && looser_ind < looser_change_t.size()) {

        ld curT = max(leader_change_t[leader_ind].first, looser_change_t[looser_ind].first);
        ld curDist = getDist(velo[leader_change_t[leader_ind].second], velo[looser_change_t[looser_ind].second], curT);
        if (minTimeAndDist.second > curDist) {
            minTimeAndDist = {curT, curDist};
        }

        if (leader_ind == leader_change_t.size() - 1 && looser_ind == looser_change_t.size() - 1) {
            break;
        }

        if (leader_ind == leader_change_t.size() - 1) {
            looser_ind++;
        } else if (looser_ind == looser_change_t.size() - 1) {
            leader_ind++;
        } else {
            if (leader_change_t[leader_ind + 1].first < looser_change_t[looser_ind + 1].first) {
                leader_ind++;
            } else if(abs(leader_change_t[leader_ind + 1].first - looser_change_t[looser_ind + 1].first) < 0.000001) {
                leader_ind++;
                looser_ind++;
            } else {
                looser_ind++;
            }
        }
    }

    cout << minTimeAndDist.first << ' ' << minTimeAndDist.second << endl;

    return 0;
}
