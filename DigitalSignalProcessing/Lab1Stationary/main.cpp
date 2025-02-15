#include <iostream>
#include <vector>
#include <fstream>
#include <locale>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

typedef long long ll;
typedef long double ld;

typedef ld metricFunc(vector<ld>, ll, ll);

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

struct MetricProvider {
    metricFunc *compute;
    string name;
};

vector<ld> readDataFromFile(const string &path) {
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

vector<pair<ll, vector<ll>>> readProbDataFromFile(const string &path) {
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

void printVec(const vector<ld> &dataVec, const string &delimiter) {
    for (ld x: dataVec)
        cout << x << delimiter;
}

void printVec(const vector<ld> &dataVec) {
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

vector<ld> getBlockMetricVector(const vector<ld> &dataVec, ll blockSize, const MetricProvider &metricProvider) {
    ll totalBlocks = getTotalBlocksCount(dataVec.size(), blockSize);
    vector<ld> blockMetrics(totalBlocks);
    for (ll blockInd = 0; blockInd < totalBlocks; blockInd++) {
        ld computedMetric = metricProvider.compute(dataVec, blockInd * blockSize, (blockInd + 1) * blockSize - 1);
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

// Средне квадратическое отклонение
ld squaredDistance(vector<ld> dataVec, ll lPos, ll rPos) {
    ll n = (rPos - lPos + 1);
    ld avg = 0.0l;
    if (lPos >= dataVec.size())
        return 0.0l;
    for (ll i = lPos; i <= min(rPos, (ll) dataVec.size() - 1); i++)
        avg += dataVec[i];
    avg /= n;

    ld res = 0.0l;
    for (ll i = lPos; i <= min(rPos, (ll) dataVec.size() - 1); i++)
        res += pow(dataVec[i] - avg, 2);

    return sqrt(res / (n));
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

ll sumVec(const vector<ll> &dataVec) {
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

    ld dy, dx, k, b;
    ll i, inext;

    if (probVec[lessN2Ind].first < N2 || lessN2Ind == probVec.size() - 1) {
            inext = lessN2Ind;
            i = lessN2Ind - 1;
    } else {
        inext = lessN2Ind + 1;
        i = lessN2Ind;
    }

    dy = probVec[inext].second[probInd] - probVec[i].second[probInd];
    dx = probVec[inext].first - probVec[i].first;
    k = (ld) dy / dx;
    b = ((probVec[inext].second[probInd] - k * probVec[inext].first) + (probVec[i].second[probInd] - k * probVec[i].first)) / 2.0l;
    return (ll) (k * N2 + b);

}

vector<stationery_gip>
getSeriesGip(const vector<ld> &dataVec, ll blockCntFrom, ll blockCntTo, const MetricProvider &metricProvider) {
    vector<pair<ll, vector<ll>>> seriesProbVec = readProbDataFromFile(
            "..\\seriesProb.txt");

    vector<stationery_gip> result;
    for (ll blockSize = blockCntFrom; blockSize <= blockCntTo; blockSize++) {
        vector<ld> metricBlockVec = getBlockMetricVector(dataVec, blockSize, metricProvider);
        vector<pair<ll, pair<ll, ll>>> seriesVec = getSeries(metricBlockVec);

        ll N2 = metricBlockVec.size() / 2;
        stationery_gip gip;
        gip.name = ("S-" + to_string(blockSize));
        gip.b = blockSize;
        gip.N = N2;
        gip.metricCnt = seriesVec.size();
        gip.p095 = getProb(seriesProbVec, N2, 2);
        gip.p005 = getProb(seriesProbVec, N2, 3);
        gip.p099 = getProb(seriesProbVec, N2, 0);
        gip.p001 = getProb(seriesProbVec, N2, 5);

        result.push_back(gip);
    }
    return result;
}

vector<stationery_gip> getInverseGip(vector<ld> dataVec, ll blockCntFrom, ll blockCntTo, const MetricProvider &metricProvider) {
    vector<pair<ll, vector<ll>>> inverseProbVec = readProbDataFromFile(
            "..\\inverseProb.txt");

    vector<stationery_gip> result;
    for (ll blockSize = blockCntFrom; blockSize <= blockCntTo; blockSize++) {
        vector<ld> metricBlockVec = getBlockMetricVector(dataVec, blockSize, metricProvider);
        vector<ll> inverseVec = getInverseVector(metricBlockVec);

        ll N = metricBlockVec.size();
        stationery_gip gip;
        gip.name = ("I-" + to_string(blockSize));
        gip.b = blockSize;
        gip.N = N;
        gip.metricCnt = sumVec(inverseVec);
        gip.p095 = getProb(inverseProbVec, N, 2);
        gip.p005 = getProb(inverseProbVec, N, 3);
        gip.p099 = getProb(inverseProbVec, N, 0);
        gip.p001 = getProb(inverseProbVec, N, 5);

        result.push_back(gip);
    }
    return result;
}

template<typename T>
void printElement(T t) {
    cout << left << setw(8) << setfill(' ') << t;
}

void printTableInverseGip(const vector<stationery_gip> &inverseGipVec) {
    cout << left << "----------------------- TABLE INVERSE -----------------------" << '\n';
    printElement("NAME");
    printElement("b");
    printElement("N");
    printElement("Cnt");
    printElement("a095");
    printElement("a005");
    printElement("a099");
    printElement("a001");
    cout << '\n';

    for (const stationery_gip &gip: inverseGipVec) {
        printElement(gip.name);
        printElement(gip.b);
        printElement(gip.N);
        printElement(gip.metricCnt);
        printElement(gip.p095);
        printElement(gip.p005);
        printElement(gip.p099);
        printElement(gip.p001);
        cout << '\n';
    }
}

void printTableSeriesGip(const vector<stationery_gip> &seriesGipVec) {
    cout << left << "----------------------- TABLE SERIES -----------------------" << '\n';
    printElement("NAME");
    printElement("b");
    printElement("N/2");
    printElement("Cnt");
    printElement("a095");
    printElement("a005");
    printElement("a099");
    printElement("a001");
    cout << '\n';

    for (const stationery_gip &gip: seriesGipVec) {
        printElement(gip.name);
        printElement(gip.b);
        printElement(gip.N);
        printElement(gip.metricCnt);
        printElement(gip.p095);
        printElement(gip.p005);
        printElement(gip.p099);
        printElement(gip.p001);
        cout << '\n';
    }
}


bool isAckSeries(ll value, ll left, ll right) {
    return left < value && value <= right;
}

bool isAckInverse(ll value, ll left, ll right) {
    return left < value && value < right;
}

void printTableAckGip(const vector<stationery_gip> &seriesGipVec, const vector<stationery_gip> &inverseGipVec) {
    cout << left << "----------------------- TABLE 3 -----------------------" << '\n';
    printElement("NAME");
    printElement("b");
    printElement("phi1");
    printElement("phi2");
    printElement("psi1");
    printElement("psi2");
    cout << '\n';

    for (ll i = 0; i < min(seriesGipVec.size(), inverseGipVec.size()); i++) {
        auto& series_gip = seriesGipVec[i];
        auto& inverse_gip = inverseGipVec[i];

        int isAckSeries1 = isAckSeries(series_gip.metricCnt, series_gip.p095, series_gip.p005);
        int isAckSeries2 = isAckSeries(series_gip.metricCnt, series_gip.p099, series_gip.p001);
        int isAckInverse1 = isAckSeries(inverse_gip.metricCnt, inverse_gip.p095, inverse_gip.p005);
        int isAckInverse2 = isAckSeries(inverse_gip.metricCnt, inverse_gip.p099, inverse_gip.p001);

        printElement("SI-" + to_string(series_gip.b));
        printElement(series_gip.b);
        printElement(isAckSeries1);
        printElement(isAckSeries2);
        printElement(isAckInverse1);
        printElement(isAckInverse2);

        cout << '\n';
    }
}

void compute(const vector<ld>& dataVec, ll blockSizeFrom, ll blockSizeTo, const vector<MetricProvider> &metricProviders) {
    for (const auto &metric: metricProviders) {
        cout << "------------------------------------------------------------------------------" << '\n';
        cout << "SIGNAL ANALYZE BY METRIC: " << metric.name << '\n';
        vector<stationery_gip> seriesGipVec = getSeriesGip(dataVec, blockSizeFrom, blockSizeTo, metric);
        vector<stationery_gip> inverseGipVec = getInverseGip(dataVec, blockSizeFrom, blockSizeTo, metric);

        printTableSeriesGip(seriesGipVec);
        printTableInverseGip(inverseGipVec);
        printTableAckGip(seriesGipVec, inverseGipVec);
        cout << "------------------------------------------------------------------------------" << '\n';
    }
}

// 3 ВАРИАНТ
int main() {
    vector<ld> dataVec = readDataFromFile("..\\N31.txt");

    cout << "Print block sizes (from, to): ";
    d_cl(blockCntFrom);
    d_cl(blockCntTo);

    MetricProvider avgSq;
    avgSq.name = "Average Square";
    avgSq.compute = &averageSquare;

    MetricProvider sqrtDist;
    sqrtDist.name = "Squared Distance";
    sqrtDist.compute = &squaredDistance;

    compute(dataVec, blockCntFrom, blockCntTo, {avgSq, sqrtDist});
    return 0;
}
