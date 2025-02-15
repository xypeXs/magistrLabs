#include <iostream>
#include <vector>
#include <fstream>
#include <locale>
#include <cmath>
#include <algorithm>

using namespace std;

typedef long long ll;
typedef long double ld;

#define d_c(type, var) type var; cin >> var;
#define d_cl(var) d_c(ll, var);
#define d_cd(type, var) d_c(ld, var);

struct stationery_gip {
    string name;
    ll b;
    ll N;
    ll metricCnt;
    ll p095;
    ll p005;
    ll p099;
    ll p001;
};

struct gip_ack {
    string name;
    bool fi1; // series 005
    bool fi2; // series 001
    bool psi1; // inverse 005
    bool psi2; // inverse 001
};

struct Comma final : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
};

vector<ld> readDataFromFile(string path) {
    ifstream is(path);
    if (!is.is_open())
        throw;

    is.imbue(locale(locale::classic(), new Comma)); // read numbers with comma delimiters

    vector<ld> dataVec;
    ld data;
    while (is >> data) {
        dataVec.push_back(data);
    }
    is.close();
    return dataVec;
}

vector<pair<ll, vector<ll>>> readProbDataFromFile(string path) {
    ifstream is(path);
    if (!is.is_open())
        throw;

    ll N;
    vector<pair<ll, vector<ll>>> probVec;
    while (is >> N) {
        ll p099, p0975, p095, p005, p0025, p001;
        is >> p099 >> p0975 >> p095 >> p005 >> p0025 >> p001;
        probVec.push_back({N, {p099, p0975, p095, p005, p0025, p001}});
    }
    is.close();
    return probVec;
}

void printVec(vector<ld> dataVec, string delimiter) {
    for (ld x: dataVec)
        cout << x << delimiter;
}

void printVec(vector<ld> dataVec) {
    printVec(dataVec, " ");
}

ld getMedian(vector<ld> dataVec) {
    vector<ld> medianVec(dataVec);
    sort(medianVec.begin(), medianVec.end());
    if (medianVec.size() == 1)
        return medianVec[0];

    if (medianVec.size() & 1)
        return (medianVec[medianVec.size() / 2] + medianVec[medianVec.size() / 2 + 1]) / 2;

    return medianVec[medianVec.size() / 2];
}

// <-1 - меньше медианы, 1 - больше, <начало серии, конец серии>>
vector<pair<ll, pair<ll, ll>>> getSeries(vector<ld> dataVec) {
    ld median = getMedian(dataVec);
    vector<pair<ll, pair<ll, ll>>> seriesVec;
    pair<ll, ll> curSeries = {0, 0};
    ll curDiff = dataVec[curSeries.first] - median < 0 ? -1ll : 1ll;
    for (ll i = 1; i < dataVec.size(); i++)
        if (curDiff * (dataVec[i] - median) < 0) { // Знаки отличаются
            curSeries.second = i - 1;
            seriesVec.push_back({curDiff, curSeries});
            curSeries = {i, i};
            curDiff = dataVec[curSeries.first] - median < 0 ? -1ll : 1ll;
        }
    return seriesVec;
}

ll getTotalBlocksCount(ll size, ll blockSize) {
    return size / blockSize + (size % blockSize > 0);
}

vector<ld> getBlockMetricVector(vector<ld> dataVec, ll blockSize, ld (metricProvider)(vector<ld>, ll, ll)) {
    ll totalBlocks = getTotalBlocksCount(dataVec.size(), blockSize);
    vector<ld> blockMetrics(totalBlocks);
    for (ll blockInd = 0; blockInd < totalBlocks; blockInd++) {
        ld computedMetric = metricProvider(dataVec, blockInd * blockSize, (blockInd + 1) * blockSize - 1);
        blockMetrics[blockInd] = (computedMetric);
    }
    return blockMetrics;
}

ld averageSquare(vector<ld> dataVec, ll lPos, ll rPos) {
    ld res = 0;
    if (lPos >= dataVec.size())
        return 0.0l;
    for (ll i = lPos; i <= min(rPos, (ll) dataVec.size() - 1); i++)
        res += dataVec[i];

    return 1.0l / (rPos - lPos + 1) * res;
}

vector<ll> getInverseVector(vector<ld> dataVec) {
    vector<ll> inverseVec(dataVec.size() - 1);
    for (ll i = 0; i < dataVec.size() - 1; i++) {
        ll inverseCnt = 0;
        for (ll j = i + 1; j < dataVec.size(); j++) {
            if (dataVec[j] < dataVec[i])
                inverseCnt++;
        }
        inverseVec[i] = inverseCnt;
    }
    return inverseVec;
}

ll sumVec(vector<ll> dataVec) {
    ll sum = 0;
    for (ll x: dataVec)
        sum += x;
    return sum;
}

// 0.99 0.975 0.95 0.05 0.025 0.01
ll getProb(vector<pair<ll, vector<ll>>> probVec, ll N2, ll probInd) {
    ll lessN2Ind = 0;
    for (; lessN2Ind < probVec.size(); lessN2Ind++) {
        if (probVec[lessN2Ind].first > N2) {
            lessN2Ind--;
            break;
        }
    }
    lessN2Ind = min(lessN2Ind, (ll) probVec.size() - 1);

    if (probVec[lessN2Ind].first == N2)
        return probVec[lessN2Ind].second[probInd];

    ld dy, dx, k;

    if (probVec[lessN2Ind].first < N2) {
        if (lessN2Ind >= probVec.size()) {
            dy = probVec[lessN2Ind].second[probInd] - probVec[lessN2Ind - 1].second[probInd];
            dx = probVec[lessN2Ind].first - probVec[lessN2Ind - 1].first;
        } else {
            dy = probVec[lessN2Ind + 1].second[probInd] - probVec[lessN2Ind].second[probInd];
            dx = probVec[lessN2Ind + 1].first - probVec[lessN2Ind].first;
        }
    } else { // Стоим на первом элементе
        dy = probVec[lessN2Ind + 1].second[probInd] - probVec[lessN2Ind].second[probInd];
        dx = probVec[lessN2Ind + 1].first - probVec[lessN2Ind].first;
    }

    k = (ld) dy / dx;
    return (ll) (k * N2);

}

vector<stationery_gip> getSeriesGip(vector<ld> dataVec, ll blockCntFrom, ll blockCntTo) {
    vector<pair<ll, vector<ll>>> seriesProbVec = readProbDataFromFile(
            "C:\\university\\magistr\\labs\\DigitalSignalProcessing\\Lab1Stationary\\seriesProb.txt");

    vector<stationery_gip> result;
    for (ll blockSize = blockCntFrom; blockSize <= blockCntTo; blockSize++) {
        vector<ld> metricBlockVec = getBlockMetricVector(dataVec, blockSize, &averageSquare);
        vector<pair<ll, pair<ll, ll>>> seriesVec = getSeries(metricBlockVec);

        ll N2 = metricBlockVec.size() / 2;
        stationery_gip gip;
        gip.name = ("SERIES_AVG_SQUARE_" + to_string(blockSize));
        gip.b = blockSize;
        gip.N = N2;
        gip.metricCnt = seriesVec.size();
        gip.p095 = getProb(seriesProbVec, seriesVec.size(), 2);
        gip.p005 = getProb(seriesProbVec, seriesVec.size(), 3);
        gip.p099 = getProb(seriesProbVec, seriesVec.size(), 0);
        gip.p001 = getProb(seriesProbVec, seriesVec.size(), 5);

        result.push_back(gip);
    }
    return result;
}

vector<stationery_gip> getInverseGip(vector<ld> dataVec, ll blockCntFrom, ll blockCntTo) {
    vector<pair<ll, vector<ll>>> inverseProbVec = readProbDataFromFile(
            "C:\\university\\magistr\\labs\\DigitalSignalProcessing\\Lab1Stationary\\inverseProb.txt");

    vector<stationery_gip> result;
    for (ll blockSize = blockCntFrom; blockSize <= blockCntTo; blockSize++) {
        vector<ld> metricBlockVec = getBlockMetricVector(dataVec, blockSize, &averageSquare);
        vector<ll> inverseVec = getInverseVector(metricBlockVec);

        ll N2 = metricBlockVec.size() / 2;
        stationery_gip gip;
        gip.name = ("INVERSE_AVG_SQUARE_" + to_string(blockSize));
        gip.b = blockSize;
        gip.N = N2;
        gip.metricCnt = inverseVec.size();
        gip.p095 = getProb(inverseProbVec, inverseVec.size(), 2);
        gip.p005 = getProb(inverseProbVec, inverseVec.size(), 3);
        gip.p099 = getProb(inverseProbVec, inverseVec.size(), 0);
        gip.p001 = getProb(inverseProbVec, inverseVec.size(), 5);

        result.push_back(gip);
    }
    return result;
}

void printTableGip(string name, vector<stationery_gip> gip_vec) {
    cout << "-----------TABLE " + name + "-----------" << '\n';
    for (stationery_gip gip : gip_vec) {
        cout << gip.name << ' '
        << gip.b << ' '
        << gip.N << ' '
        << gip.metricCnt << ' '
        << gip.p095 << ' '
        << gip.p005 << ' '
        << gip.p099 << ' '
        << gip.p001
        << '\n';
    }
}

bool isAck(ld r, ld Nb, ld a, ld Nr) {
    return r * Nb / 2
}

vector<gip_ack> getAckGip(vector<ld> blockVec, vector<stationery_gip> seriesVec, vector<stationery_gip> inverseVec) {
    ld median = getMedian(blockVec);
    for (ll i = 0; i < seriesVec.size(); i++) {
        gip_ack ack;
        ack.name = "ACK " + to_string(seriesVec[i].b);
        ack.fi1 =
    }
}

void compute(vector<ld> dataVec, ll blockSizeFrom, ll blockSizeTo) {
    vector<stationery_gip> seriesGip = getSeriesGip(dataVec, blockSizeFrom, blockSizeTo);
    vector<stationery_gip> inverseGip = getInverseGip(dataVec, blockSizeFrom, blockSizeTo);


    printTableGip("SERIES", seriesGip);
    printTableGip("INVERSE", seriesGip);
}

// 3 ВАРИАНТ
int main() {
    vector<ld> dataVec = readDataFromFile(
            "C:\\university\\magistr\\labs\\DigitalSignalProcessing\\Lab1Stationary\\HL_Makh.txt");

    cout << "Print block sizes (from, to): ";
    d_cl(blockCntFrom);
    d_cl(blockCntTo);
    compute(dataVec, blockCntFrom, blockCntTo);
    return 0;
}
